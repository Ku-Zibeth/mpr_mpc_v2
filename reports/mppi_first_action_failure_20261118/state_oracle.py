"""Frozen encoder/dynamics/terminal Q oracle, entirely offline on old NPZ files."""
import itertools
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))
import numpy as np
import torch
from omegaconf import OmegaConf
from analyze import OLD, R, load, dump, table, split_of
from tools.first_action_study import file_sha, tensor_hash
from tools.first_action_failure import ranking, cluster_bootstrap, paired_origins, correlation, isolated_state_inputs
from mpr_mpc._bootstrap import bootstrap
bootstrap()
from common.world_model import WorldModel
from common import math as td_math
from tdmpc2 import TDMPC2
import evaluate_model_suite as suite


METRICS = ['pair_accuracy', 'spearman', 'kendall', 'top1_regret', 'baseline_accuracy',
           'positive_identification_rate', 'baseline_gain_mae', 'selected_gain']


@torch.no_grad()
def main():
    p = load(OLD / 'protocol.json')
    assert file_sha(Path(p['checkpoint'])) == p['checkpoint_sha256']
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision('high')
    raw = OmegaConf.load(p['source_environment_config'])
    raw.restore_checkpoint_runtime = True
    raw.restore_mppi_runtime = False
    _, cfg = suite._make_cfg(raw, Path(p['checkpoint']).parent.parent.name, Path(p['checkpoint']), [1])
    cfg.obs_shape, cfg.action_dim, cfg.episode_length = {'state': (259,)}, 2, 1000
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = WorldModel(cfg).to(device).eval().requires_grad_(False)
    agent = SimpleNamespace(model=model, cfg=cfg)
    TDMPC2.load(agent, {'model': torch.load(p['checkpoint'], map_location='cpu', weights_only=False)['model']})
    original_hash = tensor_hash(model)
    details, group_rows, pair_errors, arrays_out, control = [], [], [], {}, []
    cached_discrepancies, encoder_discrepancies, cached_q_checks = [], [], []

    def value(z, a, members):
        qs = td_math.two_hot_inv(model.Q(z, a, None, return_type='all'), cfg).squeeze(-1)
        return qs[list(members)].mean(0).cpu().numpy()

    for folder in sorted((OLD / 'origins').iterdir()):
        if not (folder / 'catalog.json').exists():
            continue
        c = {k: a.copy() for k, a in np.load(folder / 'catalog.npz').items()}
        meta = load(folder / 'catalog.json')
        scene = meta['scene']
        before = int(folder.name.split('_')[1])
        b = len(meta['representatives'])
        control_a, control_b = isolated_state_inputs(c['F2'], None, None, 'current_qb')
        control.append(dict(scene_id=scene, origin_id=folder.name, equal=bool(np.array_equal(control_a, control_b)),
                            explanation='Q_B takes CURRENT535+a0. Both valid conditions have identical input; swapping in z_future would change time/target and create hybrid535.'))
        for h in p['horizons']:
            records, branches = [], []
            for rep in meta['representatives']:
                f = OLD / 'branches' / folder.name / f"H{h}_{rep['candidate_id']}_baseline.json"
                if not f.exists():
                    break
                rec = load(f)
                with np.load(ROOT / rec['source']) as archive:
                    data = {k: v.copy() for k, v in archive.items()}
                records.append(rec)
                branches.append(data)
            if len(records) != b:
                continue
            actual_latents = []
            for data in branches:
                # Encode ONLY observations that really exist. No padded future state after done.
                obs = torch.as_tensor(data['observations'][:h+1], device=device)
                actual_latents.append(model.encode(obs, None).cpu().numpy())
            encoder_discrepancies.extend(float(np.max(np.abs(z[0] - c['z0']))) for z in actual_latents)
            z = torch.as_tensor(c['z0'], device=device).reshape(1, -1).expand(b, -1)
            action = torch.as_tensor(c['actions'], device=device)
            prefix, terminated = np.zeros(b), np.zeros(b, bool)
            pred_rewards, teacher_rewards, predicted_z = [], [], [z.cpu().numpy().copy()]
            for t in range(h):
                reward = td_math.two_hot_inv(model.reward(z, action[:, t], None), cfg).squeeze(-1).cpu().numpy()
                pred_rewards.append(reward)
                prefix += p['gamma'] ** t * (~terminated) * reward
                z = model.next(z, action[:, t], None)
                predicted_z.append(z.cpu().numpy().copy())
                if cfg.episodic:
                    terminated |= (model.termination(z, None).squeeze(-1).cpu().numpy() > .5)
            zp = z.cpu().numpy()
            pred_rewards = np.stack(pred_rewards, 1)
            predicted_z = np.stack(predicted_z, 1)
            pair = tuple(int(v) for v in c['actual_final_pair'])
            valid = np.array([len(data['actions']) >= h for data in branches])
            za = np.full_like(zp, np.nan)
            for i in np.flatnonzero(valid):
                za[i] = actual_latents[i][h]
            # H10 already has exact historical endpoints/rewards/masks. Reuse
            # them rather than changing a sensitive Q by a different rollout
            # batch shape. Keep the same full258 Q batch for actual-state swaps.
            if h == 10:
                recomputed_q = value(z, action[:, h], pair)
                recomputed_score = prefix + p['gamma']**h * (~terminated) * recomputed_q
                ids = c['pool_indices']
                cached_discrepancies.append(float(np.max(np.abs(recomputed_score-c['scores'][ids]))))
                zp = c['terminal_latents'][ids].copy()
                predicted_z[:, -1] = zp
                prefix = c['prefix'][ids].copy()
                discount = c['discount'][ids].copy()
                terminated = discount == 0
                qpred = c['q_members'][ids][:, pair].mean(1)
                full_z = torch.as_tensor(c['terminal_latents'], device=device)
                full_a = torch.as_tensor(c['pool'][:, h], device=device)
                check_q = value(full_z, full_a, pair)
                cached_q_checks.append(float(np.max(np.abs(check_q-c['q_members'][:,pair].mean(1)))))
            else:
                qpred = value(z, action[:, h], pair)
                discount = p['gamma'] ** h * (~terminated)
            qreal = np.full(b, np.nan)
            if valid.any():
                if h == 10:
                    swapped = full_z.clone()
                    swapped[torch.as_tensor(ids[valid], device=device)] = torch.tensor(za[valid], device=device)
                    qreal[valid] = value(swapped, full_a, pair)[ids[valid]]
                else:
                    qreal[valid] = value(torch.tensor(za[valid], device=device), action[valid, h], pair)
            actual_prefix = np.array([np.dot(data['rewards'][:h], p['gamma'] ** np.arange(min(h, len(data['rewards'])))) for data in branches])
            scores = dict(J00_predicted=prefix + discount*qpred,
                          J01_actual_latent_same_predicted_mask=prefix + discount*qreal,
                          J10_actual_reward=actual_prefix + discount*qpred,
                          J11_actual_reward_actual_latent_same_mask=actual_prefix + discount*qreal)
            native_continues = np.array([len(data['actions']) >= h and not data['done'][h-1] for data in branches])
            # Joint oracle deliberately changes termination AND state/reward. Not a pure latent ablation.
            joint = actual_prefix.copy()
            joint[native_continues] += p['gamma'] ** h * qreal[native_continues]
            scores['joint_actual_reward_state_native_terminal'] = joint
            if h == 10:
                assert np.allclose(scores['J00_predicted'], c['scores'][ids], atol=3e-5, rtol=0)
                scores['J00_predicted'] = c['scores'][ids].copy()
            y = np.array([r['actual_return'] for r in records])
            common = dict(scene_id=scene, origin_id=folder.name, horizon=h, split=split_of(scene, p), q_members=list(pair))
            key = f'{folder.name}_H{h}'
            arrays_out[key+'_predicted'] = predicted_z
            arrays_out[key+'_actual_endpoint'] = za
            arrays_out[key+'_endpoint_available'] = valid
            for condition, score in scores.items():
                # Pure state/reward effects are compared on IDENTICAL endpoint-available candidates.
                for scope, selected in [('matched_endpoint_subset', valid), ('all_including_early_terminal', np.ones(b, bool))]:
                    if scope == 'all_including_early_terminal' and condition not in ['J00_predicted', 'joint_actual_reward_state_native_terminal']:
                        continue
                    ids = np.flatnonzero(selected)
                    if len(ids) < 2:
                        continue
                    rr = ranking(score[ids], y[ids])
                    baseline_available = bool(0 in ids)
                    if not baseline_available:
                        for name in ['baseline_accuracy', 'positive_identification_rate', 'baseline_gain_mae', 'selected_gain']:
                            rr[name] = None
                    group_rows.append(common | dict(condition=condition, scope=scope, candidates=len(ids),
                        baseline_available=baseline_available, missing_endpoints=int((~valid).sum())) | rr)
            for i, (rep, data, real_z) in enumerate(zip(meta['representatives'], branches, actual_latents)):
                n = min(h, len(data['actions']))
                rz = real_z[n]
                masked_pred = predicted_z[i, n]
                mse = float(np.mean((masked_pred - rz)**2))
                cosine = float(np.dot(masked_pred, rz)/(np.linalg.norm(masked_pred)*np.linalg.norm(rz)))
                teacher = td_math.two_hot_inv(model.reward(torch.tensor(real_z[:n], device=device), action[i, :n], None), cfg).squeeze(-1).cpu().numpy()
                actual_reward = data['rewards'][:n]
                details.append(common | dict(candidate_id=rep['candidate_id'], effective_prefix_steps=n,
                    requested_endpoint_available=bool(valid[i]), native_terminal_at_or_before_h=not bool(native_continues[i]),
                    latent_error_step=n, latent_mse=mse, latent_l2=float(np.linalg.norm(masked_pred-rz)), latent_cosine=cosine,
                    predicted_prefix=float(prefix[i]), actual_prefix=float(actual_prefix[i]),
                    prefix_reward_error=float(prefix[i]-actual_prefix[i]),
                    rollout_reward_mae=float(np.abs(pred_rewards[i, :n]-actual_reward).mean()),
                    teacher_forced_reward_mae=float(np.abs(teacher-actual_reward).mean()),
                    predicted_terminal_q=float(qpred[i]), actual_latent_terminal_q=float(qreal[i]) if valid[i] else None,
                    actual_continuation_after_requested_prefix=float(data['rtg'][h]) if len(data['actions']) >= h else 0.,
                    actual_total_return=float(y[i]), predicted_terminal_mask=bool(terminated[i]),
                    **{name:float(score[i]) if np.isfinite(score[i]) else None for name,score in scores.items()},
                    array_source='state_oracle_arrays.npz::'+key,
                    endpoint_note='native terminated early; no actual z_H exists' if not valid[i] else 'same requested H'))
            for i, j in itertools.combinations(np.flatnonzero(valid), 2):
                delta_error = float(np.linalg.norm((zp[i]-zp[j])-(za[i]-za[j])))
                gain = float(y[i]-y[j])
                if abs(gain) <= 1.:
                    continue
                error = float(.5-.5*np.sign((scores['J00_predicted'][i]-scores['J00_predicted'][j])*gain))
                pair_errors.append(common | dict(candidate_i=int(i), candidate_j=int(j),
                    relative_dynamics_error_l2=delta_error, ranking_error_credit=error,
                    absolute_actual_return_gap=abs(gain)))
            print('state-oracle', folder.name, 'H', h, 'endpoints', int(valid.sum()), '/', b, flush=True)
    assert original_hash == tensor_hash(model)
    np.savez_compressed(R / 'state_oracle_arrays.npz', **arrays_out)
    table('predicted_vs_actual_state.csv', details)
    table('state_oracle_origin_ranking.csv', group_rows)
    table('dynamics_pair_errors.csv', pair_errors)
    summaries, comparisons = {}, {}
    for split in ['fit', 'inner_stop', 'seen_validation', 'all_seen']:
        for h in p['horizons']:
            for scope in ['matched_endpoint_subset', 'all_including_early_terminal']:
                subset = [r for r in group_rows if r['horizon'] == h and r['scope'] == scope and (split == 'all_seen' or r['split'] == split)]
                for condition in sorted({r['condition'] for r in subset}):
                    rows = [r for r in subset if r['condition'] == condition]
                    summaries[f'{split}/H{h}/{scope}/{condition}'] = cluster_bootstrap(rows, METRICS)
                    if condition != 'J00_predicted':
                        base = [r for r in subset if r['condition'] == 'J00_predicted']
                        paired = paired_origins(base, rows)
                        comparisons[f'{split}/H{h}/{scope}/{condition}'] = cluster_bootstrap(paired, [
                            'pair_accuracy_improvement', 'spearman_improvement', 'kendall_improvement',
                            'regret_improvement', 'baseline_accuracy_improvement', 'positive_identification_rate_improvement'])
    dump('state_oracle_ranking.json', dict(summaries=summaries, paired_improvements=comparisons,
        current_qb_control=control, pure_state_condition='J01; identical prefix reward, discount, predicted terminal mask and fixed candidate terminal action/Q pair.',
        missing_endpoint_policy='All early terminated labels remain in main results and joint-oracle full-set comparison. Pure state-only analysis reports matched endpoint subset separately; no invented z_H.',
        interpretation='Terminal Q has its original imagined-latent training distribution. Actual encoder latents are a diagnostic input shift, not a fair retrained model or online improvement.',
        current_qb_warning='Replacing z_t within original535 by z_H would change input time and target and leave stale features; not performed. Q_B has no candidate future-dynamics input.'))
    error_summary, associations = {}, {}
    error_keys = ['latent_mse', 'latent_l2', 'latent_cosine', 'prefix_reward_error', 'rollout_reward_mae', 'teacher_forced_reward_mae']
    for h in p['horizons']:
        rows = [r for r in details if r['horizon'] == h and r['requested_endpoint_available']]
        # First average candidates inside each origin, then resample maps.
        origin_summary = []
        for origin in sorted({r['origin_id'] for r in rows}):
            rr = [r for r in rows if r['origin_id'] == origin]
            origin_summary.append(dict(scene_id=rr[0]['scene_id'], **{k:float(np.mean([r[k] for r in rr])) for k in error_keys}))
        error_summary[f'H{h}'] = cluster_bootstrap(origin_summary, error_keys)
        rr = []
        for origin in sorted({r['origin_id'] for r in pair_errors if r['horizon'] == h}):
            local = [r for r in pair_errors if r['horizon'] == h and r['origin_id'] == origin]
            rr.append(dict(scene_id=local[0]['scene_id'],
                error_vs_rank_error_spearman=correlation([r['relative_dynamics_error_l2'] for r in local], [r['ranking_error_credit'] for r in local]),
                error_vs_return_gap_spearman=correlation([r['relative_dynamics_error_l2'] for r in local], [r['absolute_actual_return_gap'] for r in local])))
        associations[f'H{h}'] = cluster_bootstrap(rr, ['error_vs_rank_error_spearman', 'error_vs_return_gap_spearman'])
    dump('dynamics_error_analysis.json', dict(by_requested_horizon=error_summary, within_origin_pair_associations=associations,
        initial_encoder_max_abs_difference=max(encoder_discrepancies),
        cached_h10_recomputed_score_max_abs_difference=max(cached_discrepancies),
        cached_h10_same_shape_q_check_max_abs_difference=max(cached_q_checks),
        h10_prediction_source='EXACT stored historical endpoints, prefix and terminal mask; same258-candidate Q batch for actual endpoint replacement.',
        numeric_parity_note='Separate8-candidate recomputation changes floating-point rollout/Q results. Archived v1 measures that drift; final H10 preserves historical score exactly, not a new preferred model.',
        endpoint_missing=sum(not r['requested_endpoint_available'] for r in details),
        weights_unchanged=True, checkpoint_sha256=p['checkpoint_sha256'], device=str(device),
        physical_state_prediction='No WM XY decoder or per-step physical state trace in these branches; do not interpret latent distance as meters.',
        future_535_hybrids_created=False, no_future_input_in_online_or_training=True))
    print('Frozen state oracle complete:', len(details), 'branches; weights unchanged', flush=True)


if __name__ == '__main__':
    main()
