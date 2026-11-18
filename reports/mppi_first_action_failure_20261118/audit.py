"""Read-only audit of Phase 1 recordings; no environment execution or model fitting."""
import csv
import hashlib
import importlib.metadata as metadata
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
OLD = ROOT / 'reports/mppi_first_action_value_20261117'
MAIN = ROOT.parent / 'mpr_mpc_v2'


def read(path):
    return json.loads(path.read_text())


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def save(name, value):
    path = OUT / name
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def table(name, rows):
    with (OUT / name).open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(dict.fromkeys(k for row in rows for k in row)))
        writer.writeheader()
        writer.writerows(rows)


def git(path):
    return {name: subprocess.check_output(['git', *args], cwd=path, text=True).strip()
            for name, args in [('head', ['rev-parse', 'HEAD']), ('branch', ['branch', '--show-current']),
                               ('status', ['status', '--short']), ('log5', ['log', '-5', '--oneline'])]}


def main():
    protocol = read(OLD / 'protocol.json')
    previous_verification = read(OLD / 'verification.json')
    required = ['version_manifest.json', 'native_terminal_action_parity.csv', 'q_all_members.csv',
                'q_pair_comparison.csv', 'first_action_candidates.csv', 'first_action_returns.csv',
                'horizon_protocol_comparison.csv', 'paired_candidate_effects.csv', 'candidate_coverage.json',
                'ranking_metrics.json', 'scenario_bootstrap.json', 'data_coverage.json', 'training_manifest.json',
                'validation.json', 'timing_breakdown.csv', 'decision.json', 'code_changes.patch', 'verification.json']
    missing = [str(OLD / name) for name in required if not (OLD / name).is_file()]
    old_files = sorted(set(OLD.rglob('*')) | set((ROOT / 'reports/mppi_native_parity_20261117').rglob('*')))
    old_files = [f for f in old_files if f.is_file() and '__pycache__' not in f.parts]
    old_files.extend(ROOT / name for name in ['MPPI_FIRST_ACTION_VALUE_REPORT_20261117.md',
                                             'MPPI_NATIVE_PARITY_REPORT_20261117.md'])
    old_files.extend(ROOT / path for path in previous_verification['source_sha256'])
    inventory = [{'path': str(f), 'size_bytes': f.stat().st_size, 'sha256': sha(f)}
                 for f in sorted(set(old_files))]
    source_check = {path: sha(ROOT / path) == expected
                    for path, expected in previous_verification['source_sha256'].items()}
    distributions = {}
    for name in ['numpy', 'torch', 'metadrive-simulator', 'metadrive', 'omegaconf']:
        try:
            distributions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            pass
    save('experiment_manifest.json', dict(
        main=git(MAIN), prior_worktree=git(ROOT), remote_main=read(OUT / 'remote_main.json'),
        python=sys.version, executable=sys.executable, packages=distributions,
        checkpoint=str(protocol['checkpoint']), checkpoint_sha256=sha(Path(protocol['checkpoint'])),
        checkpoint_sha_matches=sha(Path(protocol['checkpoint'])) == protocol['checkpoint_sha256'],
        config_path=protocol['source_environment_config'],
        config_sha256=sha(Path(protocol['source_environment_config'])),
        audit_script_sha256=sha(Path(__file__)), prior_source_hash_matches=source_check,
        missing_prior_deliverables=missing, prior_tests_record=(OLD / 'tests_final.txt').read_text(),
        prompt_incomplete_after='4.3 样本独立性：检查：同一地图',
        scope='Only supplied Phase0/Phase1 integrity work; no new rollout, training, selection, or online changes.'))
    table('data_sha256.csv', inventory)

    rows, findings, origin_rows = [], [], []
    origins = {}
    for folder in sorted((OLD / 'origins').iterdir()):
        if not (folder / 'catalog.json').is_file():
            continue
        arrays = {k: v.copy() for k, v in np.load(folder / 'catalog.npz').items()}
        meta, snapshot = read(folder / 'catalog.json'), read(folder / 'origin.json')
        actions = arrays['actions'][:, 0]
        unique_raw = len({a.tobytes() for a in actions})
        unique_clipped = len({np.clip(a, -1, 1).tobytes() for a in actions})
        original_pool = arrays['pool'][:, 0]
        origin_rows.append(dict(origin_id=folder.name, scene_id=meta['scene'],
            representatives=len(actions), exact_unique=unique_raw, clipped_unique=unique_clipped,
            full_pool_size=len(original_pool), full_pool_unique_first_actions=len({a.tobytes() for a in original_pool}),
            baseline_preserved=bool(np.array_equal(arrays['actions'][0], arrays['baseline'])),
            alias_count=len(meta['aliases']),
            baseline_first_action=json.dumps(arrays['baseline'][0].tolist()),
            units='normalized MetaDrive steer/throttle [-1,1]; not SI steering/force'))
        origins[folder.name] = (arrays, meta, snapshot)

    for file in sorted((OLD / 'branches').glob('*/*.json')):
        record = read(file)
        if 'actual_return' not in record:
            continue
        arrays, meta, snapshot = origins[file.parent.name]
        i = next(i for i, c in enumerate(meta['representatives']) if c['candidate_id'] == record['candidate_id'])
        archive = ROOT / record['source']
        with np.load(archive) as data:
            actions, rewards = data['actions'], data['rewards'].astype(np.float64)
            done, terminated, truncated = data['done'], data['terminated'], data['truncated']
            h, n = record['horizon'], len(actions)
            committed = min(h, n)
            discounted = np.zeros(n + 1, np.float64)
            for t in range(n - 1, -1, -1):
                discounted[t] = rewards[t] + protocol['gamma'] * discounted[t + 1]
            prefix = float(np.dot(protocol['gamma'] ** np.arange(committed), rewards[:committed]))
            hash_match = all(record[key] == snapshot['hashes'][domain] for key, domain in [
                ('origin_observation_hash', 'observation'), ('origin_vehicle_state_hash', 'vehicle'),
                ('origin_traffic_state_hash', 'traffic'), ('origin_rng_state_hash', 'rng')])
            checks = dict(
                finite_actions=bool(np.isfinite(actions).all()), actions_in_bounds=bool((np.abs(actions) <= 1).all()),
                finite_rewards=bool(np.isfinite(rewards).all()), first_action_exact=bool(np.array_equal(actions[0], arrays['actions'][i, 0])),
                full_committed_prefix_exact=bool(np.array_equal(actions[:committed], arrays['actions'][i, :committed])),
                baseline_exact=bool(np.array_equal(record['baseline_first_action'], arrays['baseline'][0])),
                observations_count=len(data['observations']) == n + 1,
                initial_observation_exact=bool(np.array_equal(data['observations'][0], snapshot['observation'])),
                initial_latent_equals_current535=bool(np.array_equal(arrays['z0'], arrays['state535'][:512])),
                finite_initial_latent=bool(np.isfinite(arrays['z0']).all()),
                done_union_exact=bool(np.array_equal(done, np.logical_or(terminated, truncated))),
                only_last_done=bool(done[-1] and not done[:-1].any()),
                rtg_recomputed=bool(np.allclose(discounted, data['rtg'], rtol=0, atol=1e-8)),
                json_return_matches=abs(discounted[0] - record['actual_return']) < 1e-8,
                first_reward_time_index=abs(rewards[0] - record['actual_first_reward']) < 1e-8,
                episode_sum_matches=abs(rewards.sum() - record['episode_return']) < 1e-8,
                continuation_after_first_matches=abs(discounted[1] - record['actual_continuation_return']) < 1e-8,
                prefix_decomposition_matches=abs(discounted[0] - prefix - protocol['gamma'] ** committed * discounted[committed]) < 1e-8,
                native_terminated_matches=bool(terminated[-1]) == record['native_terminated'],
                native_truncated_matches=bool(truncated[-1]) == record['native_truncated'],
                origin_hashes_match=hash_match, physical_prefix_remapped_zero=record['physical_action_mapping_count'] == 0,
                weights_recorded_unchanged=record['weights_unchanged'],
                feedback_step_count=(record['takeover'] != 'baseline' or record['feedback_lattice_sac_steps'] == max(0, n - h)))
            failures = [k for k, v in checks.items() if not v]
            if failures:
                findings.append(dict(branch=str(file), failed_checks=failures))
            baseline = arrays['baseline'][0]
            candidate = arrays['actions'][i, 0]
            index = int(arrays['pool_indices'][i])
            rows.append(dict(scene_id=record['scenario_id'], origin_id=record['origin_id'], candidate_id=record['candidate_id'],
                horizon=h, repeat_seed=record['policy_seed'], environment_seed=record['episode_seed'],
                exact_replay=record['replay'], takeover=record['takeover'],
                split=('fit' if record['scenario_id'] in protocol['fit_scenes'] else 'inner_stop' if record['scenario_id'] in protocol['inner_stop_scenes'] else 'previously_seen_validation'),
                initial_observation=str(archive) + '::observations[0]',
                initial_latent=str(OLD / 'origins' / record['origin_id'] / 'catalog.npz') + '::z0',
                baseline_action=json.dumps(baseline.tolist()), candidate_action=json.dumps(candidate.tolist()),
                executed_action=json.dumps(actions[0].tolist()), normalized_steering_delta=float(candidate[0] - baseline[0]),
                normalized_throttle_delta=float(candidate[1] - baseline[1]), physical_control_delta_l2=float(np.linalg.norm(candidate - baseline)),
                physical_units='MetaDrive normalized physical-control command, not residual',
                measured_SI_steering_or_force_delta=None, predicted_score=float(arrays['scores'][index]),
                actual_reward_sequence=str(archive) + '::rewards', actual_discounted_return=float(discounted[0]),
                actual_prefix_discounted_return=prefix, actual_continuation_after_fixed_prefix=float(discounted[committed]),
                actual_continuation_after_first=float(discounted[1]), success=record['success'], crash=record['crash'],
                out_of_road=record['out_of_road'], route_completion=record['route_completion'],
                terminated=bool(terminated[-1]), truncated=bool(truncated[-1]), native_done=bool(done[-1]),
                prefix_steps_actually_executed=committed, terminated_before_requested_prefix_end=n < h,
                episode_length=n, termination_reason=record['termination_reason'],
                per_step_crash_or_offroad_labels_available=False, origin_hashes_match=hash_match,
                checks_passed=not failures, failed_checks=json.dumps(failures),
                raw_json=str(file), raw_npz=str(archive)))
    table('branch_index.csv', rows)
    table('candidate_integrity.csv', origin_rows)
    counts = Counter(f"H{r['horizon']}/{r['takeover']}" for r in rows)
    unique = {(r['scene_id'], r['origin_id'], r['candidate_id'], r['horizon'], r['takeover']) for r in rows}
    summary = dict(branches=len(rows), branch_counts=dict(counts), origins=len(origins),
        maps=len({r['scene_id'] for r in rows}), unique_without_exact_replays=len(unique),
        replay_branches=sum(r['exact_replay'] for r in rows), failed_branches=findings,
        missing_prior_deliverables=missing, prior_source_hash_matches=source_check,
        all_checked_records_passed=not findings and not missing and all(source_check.values()),
        independently_validated_new_model=False,
        limitations=[
            'Actual post-clipping wheel angle/force and per-step ego physical states were not recorded; SI action mapping cannot be certified from normalized commands alone.',
            'Per-step collision/offroad/success flags were not archived; first accident time cannot be reconstructed from episode-level flags.',
            'No independent traffic-stream repeats; replay seed is not a new independent map.',
            'H2/H4/H10 exist only for six pilot maps. H1 has twelve development and six previously exposed validation maps.',
            'The prior validation maps are now seen data for attribution, never a fresh independent validation set.',
            'actual_continuation_return in old branch JSON means RTG after the first step, including remaining forced actions when H>1; the index distinguishes both continuation definitions.',
            'The user prompt is truncated at section4.3. No unseen later experimental design or acceptance thresholds are assumed.'])
    save('integrity_audit.json', summary)
    print(json.dumps(dict(branches=len(rows), maps=summary['maps'], counts=dict(counts),
                          failures=len(findings), all_checked_records_passed=summary['all_checked_records_passed']), indent=2), flush=True)


if __name__ == '__main__':
    main()
