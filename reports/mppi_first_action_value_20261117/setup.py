"""Freeze new scenes, representative selection and evidence gates before outcomes."""
import subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from tools.first_action_study import *
from tools.benchmark_mppi import source_fingerprint

old=load(SOURCE/'reports/mppi_selection_20261117/protocol.json');rng=np.random.default_rng(2026111718)
pilot=rng.choice(np.arange(90000,91000),6,replace=False).tolist();extra=rng.choice(np.arange(92000,93000),6,replace=False).tolist();validation=rng.choice(np.arange(94000,95000),6,replace=False).tolist()
REPORT.mkdir(exist_ok=True);NATIVE.mkdir(exist_ok=True)
assert not (REPORT/'protocol.json').exists()
p=dict(base_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True,cwd=ROOT).strip(),branch=subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip(),
    checkpoint=old['checkpoint'],checkpoint_sha256=old['checkpoint_sha256'],source_environment_config=old['source_environment_config'],source_root=str(SOURCE),
    pilot_scenes=pilot,extra_development_scenes=extra,validation_scenes=validation,before_steps=[50,150],horizons=[1,2,4,10],gamma=.995,
    fit_scenes=(pilot+extra)[:8],inner_stop_scenes=(pilot+extra)[8:],sealed_previous_scenes=old['validation_scenarios'],
    candidate_roles=['baseline','mppi_selected','best_nonbaseline','q75','q50','q25','worst_nonbaseline','fixed_gaussian'],
    candidate_selection='One original64x4 Gaussian/AR1 plan perorigin; no actualreturn selection. Common capturedfinalQpair rescore for quantiles. Deduplicate exact firstaction bytes, preservefullpool/droppedroles.',
    std='unchanged .04/.12, min .01/.03, max .08/.25',rho=[.8,.8],samples=64,elites=8,iterations=4,temperature=.5,
    baseline='Original deterministic Lattice+SAC, residualON, MPPIOFF everypost-prefix control tick; no second action mapping',
    pilot_gate='Expand development collection if ANY measuredH1candidate gain>1 (fixed tolerance), as requested. Stable positivecoverage statistics are a separate training gate, not an extra collectionapproval.',
    training_gate='Only train if H1measuredoracleGain sceneCI95lower>0 and positive>1 gain at >=half reachabledevelopmentmaps AND no existing scorer has reliable positive selectedgain. No validationlabels read beforeONE model fixed.',
    value_gate='Independentnew6map accuracyCI95lower>.5,SpearmanCI95lower>0,pairedRegretimprovementCI95lower>0; multipleorigins/maps; allmustpass. Otherwise NO onlinecode.',
    model_plan='F1/F2/F3, independentridge vs two-hiddenlayerMLP[64,32]; scene-balancedHuber+pairwise. Fixedseeds, onepredeclaredregularization, no hyperparametersearch. Pairtolerance1. LabelsH1ONLY. Priorheadsneverloaded.',
    primary_q='actual capturedfinalpair forH10score; currentstateQ andallreducers evaluatedoffline; native sampledterminal onefixed draw (plus100Qpair draws separately)',
    native_takeover='first2pilotmaps atbefore50; commonfirstBaselinephysicalaction, then originalTDActor orFULLTD-MPC2act(mpcTrue), comparecachedH1SAC. Actor andMPC keptseparate. Native512/64/6 budget unchanged, notLocalMPPI64/8/4.',
    rng='All6global/engine/traffic RNGrecorded+strictoriginhashes. Independentprocessperbranch. No irrelevantseedreplications. Oneexactduplicate verifiesdeterministicB; randomtraffic=false kept unchanged.',
    return_tolerance=1.,bootstrap_samples=2000,bootstrap_seed=1,validation_opened=False,formal100=False,default_online_changes=False)
assert p['base_commit']=='27f8131d95a39bfe2f4a95d101fef1774903b907';assert file_sha(p['checkpoint'])==p['checkpoint_sha256']
dump(REPORT/'protocol.json',p)
version=dict(head=p['base_commit'],origin_main=subprocess.check_output(['git','rev-parse','origin/main'],cwd=ROOT,text=True).strip(),branch=p['branch'],worktree=str(ROOT),original_main=str(SOURCE),
    github_main_verified_via_connector='27f8131d95a39bfe2f4a95d101fef1774903b907',
    fetch='Requested gitfetch SSH stalled; terminatedownfetch; HTTPSdirect andlocal7897proxy timedout35s. GitHubconnector fetch_commit(main) independently confirms27f8131; no newerremote claimed.',
    checkpoint_sha256=p['checkpoint_sha256'],runtime_fingerprint=source_fingerprint(ROOT),
    log8=subprocess.check_output(['git','log','--oneline','-8'],cwd=ROOT,text=True).splitlines(),
    divergence_stat=subprocess.check_output(['git','diff','--stat','origin/main...HEAD'],cwd=ROOT,text=True),
    previous_split_audit=dict(fit_inner_stop='historical28map roots, not independent',phase1_independent_pilot=[83964,83060,83571,83797],sealed=old['validation_scenarios']),
    historical_inputs={str(f.relative_to(SOURCE)):file_sha(f) for f in [SOURCE/'MPPI_FAILURE_ATTRIBUTION_REPORT_20261117.md',SOURCE/'SAC_POLICY_MATCHED_VALUE_REPORT_20261117.md',
        SOURCE/'reports/mppi_failure_attribution_20261117/paired_statistics.json',SOURCE/'reports/mppi_failure_attribution_20261117/reward_continuation_error_decomposition.csv',
        SOURCE/'reports/mppi_failure_attribution_20261117/real_latent_q_vs_sac_continuation.csv',SOURCE/'reports/mppi_failure_attribution_20261117/q_ensemble_rank_stability.csv',
        SOURCE/'reports/sac_policy_matched_value_20261117/td_actor_value_consistency.csv',SOURCE/'reports/sac_policy_matched_value_20261117/paired_rollout_results.csv',
        SOURCE/'reports/sac_policy_matched_value_20261117/ranking_metrics.json',SOURCE/'reports/sac_policy_matched_value_20261117/decision.json']})
dump(REPORT/'version_manifest.json',version);dump(NATIVE/'version_manifest.json',version)
print('pilot',pilot,'extraDev',extra,'validationSEALED',validation,flush=True)
