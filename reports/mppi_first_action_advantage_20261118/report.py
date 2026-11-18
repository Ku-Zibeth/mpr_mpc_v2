"""Evidence-based Phase3 report/plots and preservation verification."""
import csv,json,subprocess,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
from study_io import R,ROOT,PRIOR,OLD,PHASE2,load,dump,table,sha
sys.path.insert(1,str(ROOT))
import numpy as np
from tools.first_action_failure import cluster_bootstrap,paired_origins
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def read(name):return list(csv.DictReader((R/name).open()))
def fmt(s,percent=False):
    if s['mean'] is None:return 'N/A'
    f=100 if percent else 1
    return f"{s['mean']*f:.3f} [{s['ci95'][0]*f:.3f},{s['ci95'][1]*f:.3f}]"


def plots(e,p,d):
    folder=R/'figures';folder.mkdir(exist_ok=False)
    names=['A0_frozen','A1_linear','A1_mlp','A2_linear','A2_mlp','A2_knn']
    def save(fig,name):
        fig.tight_layout();fig.savefig(folder/(name+'.png'),dpi=180);fig.savefig(folder/(name+'.pdf'));plt.close(fig)
    for metric,filename in [('pair_accuracy','absolute_vs_relative_accuracy'),('top1_regret','regret_comparison')]:
        fig,axes=plt.subplots(1,2,figsize=(12,4))
        for ax,stage in zip(axes,['lomo','inner_stop']):
            stats=[e['summaries'][f'{stage}/{n}'][metric] for n in names]
            y=np.array([s['mean'] for s in stats]);ci=np.array([s['ci95'] for s in stats]);x=np.arange(len(names))
            bars=ax.bar(x,y,color=['#aaaaaa','#87b5d7','#2878B5','#e8b08c','#C65A30','#3A8E5C'])
            if stage=='lomo':bars[0].set_hatch('//')
            ax.errorbar(x,y,yerr=np.maximum(0,[y-ci[:,0],ci[:,1]-y]),fmt='none',color='black',capsize=3)
            if metric=='pair_accuracy':ax.axhline(.5,ls='--',color='gray')
            ax.set(xticks=x,xticklabels=['A0*' if stage=='lomo' else 'A0']+[n[3:]+' '+n[:2] for n in names[1:]],ylabel=metric,
                title='8-map LOMO (*A0 is seen-fit reference)' if stage=='lomo' else '4-map inner-stop, development ONLY')
            ax.tick_params(axis='x',rotation=45)
        save(fig,filename)
    lomo=read('leave_one_map_out.csv');fig,ax=plt.subplots(figsize=(10,4));scenes=p['fit_maps']
    for name in ['A0_frozen',d['selected_A1'],d['selected_A2'],'A2_knn']:
        values=[float(next(r['pair_accuracy'] for r in lomo if r['model']==name and int(r['heldout_scene'])==s)) for s in scenes]
        ax.plot(np.arange(8),values,marker='o',ls='--' if name=='A0_frozen' else '-',label=name+(' (seen fit)' if name=='A0_frozen' else ''))
    ax.axhline(.5,color='gray',ls=':');ax.set(xticks=np.arange(8),xticklabels=scenes,ylabel='Pair accuracy',title='One withheld fit map per fold; A0 never a held-out comparator')
    ax.legend();save(fig,'leave_one_map_out')
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for ax,stage in zip(axes,['lomo','inner_stop']):
        models=['Baseline','Random','Oracle','A0_frozen',d['selected_A1'],d['selected_A2']]
        stats=[e['oracle'][f'{stage}/{n}']['gain'] for n in models];y=np.array([s['mean'] for s in stats]);ci=np.array([s['ci95'] for s in stats])
        ax.bar(np.arange(len(y)),y);ax.errorbar(np.arange(len(y)),y,yerr=np.maximum(0,[y-ci[:,0],ci[:,1]-y]),fmt='none',color='black',capsize=3)
        ax.axhline(0,color='black',lw=.8);ax.set(xticks=np.arange(len(y)),xticklabels=models,title=stage+' (A0 fit reference in LOMO)',ylabel='Actual paired H1 gain')
        ax.tick_params(axis='x',rotation=45)
    save(fig,'oracle_vs_model_gain')
    bands=['<.01','.01-.05','.05-.1','>=.1'];fig,axes=plt.subplots(1,2,figsize=(12,4))
    for ax,stage in zip(axes,['lomo','inner_stop']):
        for name in ['A0_frozen',d['selected_A1'],d['selected_A2'],'A2_knn']:
            stats=[e['magnitude'][f'{stage}/{name}/{b}']['pair_accuracy'] for b in bands]
            y=np.array([s['mean'] if s['mean'] is not None else np.nan for s in stats]);ci=np.array([s['ci95'] if s['mean'] is not None else [np.nan,np.nan] for s in stats])
            ax.errorbar(np.arange(4),y,yerr=np.maximum(0,[y-ci[:,0],ci[:,1]-y]),marker='o',capsize=3,label=name)
        ax.axhline(.5,color='gray',ls='--');ax.set(xticks=np.arange(4),xticklabels=bands,ylabel='Accuracy',xlabel='Physical command delta L2 (normalized units)',title=stage+'; Baseline retained within each subgroup');ax.legend(fontsize=8)
    save(fig,'action_delta_accuracy')


def main():
    p=load(R/'protocol.json');m=load(R/'experiment_manifest.json');e=load(R/'evaluation_results.json');d=load(R/'decision.json')
    assert not d['proceed_to_new_validation'],'Development gate passed: complete authorized NEW6-map validation before final delivery.'
    assert '82 passed' in (R/'tests_original82.log').read_text() and '16 passed' in (R/'tests_new16.log').read_text()
    assert sha(Path(p['checkpoint']))==p['checkpoint_sha256'] and sha(Path(p['A0_checkpoint']))==p['A0_sha256']
    assert sha(R/'protocol.json')==m['protocol_sha256']
    assert sha(ROOT/'config.yaml')==m['effective_default_config_sha256']
    for path,h in m['protected_source_sha256'].items():assert sha(ROOT/path)==h,path
    for x in read('input_sha256.csv'):assert sha(Path(x['path']))==x['sha256'],x['path']
    old_inventory=list(csv.DictReader((PHASE2/'data_sha256.csv').open()))
    for x in old_inventory:assert sha(Path(x['path']))==x['sha256'],x['path']
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()==p['parent_commit']
    assert subprocess.check_output(['git','diff','--name-only'],cwd=ROOT,text=True).strip()==''
    manifest=load(R/'model_manifest.json')
    assert all(sha(Path(x['checkpoint']))==x['sha256'] for x in manifest)
    comparisons={}
    origin=read('origin_results.csv')
    for row in origin:
        row['scene_id']=int(row['scene_id']);row['horizon']=int(row['horizon'])
        for k in ['pair_accuracy','spearman','kendall','top1_regret','baseline_accuracy','positive_identification_rate','model_gain','selected_negative']:
            row[k]=float(row[k]) if row.get(k) else None
    for stage in ['lomo','inner_stop']:
        for first,second in [('A1_mlp','A2_mlp'),('A2_mlp','A2_linear'),('A2_mlp','A2_knn')]:
            a=[r for r in origin if r['stage']==stage and r['model']==first];b=[r for r in origin if r['stage']==stage and r['model']==second]
            pairs=paired_origins(a,b)
            for pair,x,y in zip(pairs,sorted(a,key=lambda r:r['origin_id']),sorted(b,key=lambda r:r['origin_id'])):
                assert x['origin_id']==y['origin_id']==pair['origin_id']
                pair['model_gain_improvement']=y['model_gain']-x['model_gain']
                pair['negative_rate_increase']=y['selected_negative']-x['selected_negative']
            comparisons[f'{stage}/{second}_minus_{first}']=cluster_bootstrap(pairs,['pair_accuracy_improvement','regret_improvement','model_gain_improvement','negative_rate_increase'])
    dump('paired_statistics.json',comparisons)
    lomo=np.load(R/'lomo_predictions.npz');inner=np.load(R/'inner_stop_predictions.npz')
    linear_equal=np.array_equal(lomo['A1_linear'],lomo['A2_linear']) and np.array_equal(inner['A1_linear'],inner['A2_linear'])
    assert linear_equal
    dump('validation_status.json',dict(performed=False,reason=d['reason'],new_maps=[],new_origins=[],
        old_seen_validation_reused=False,frozen_A2_sha256=d['frozen_A2']['sha256'],online=False))
    plots(e,p,d)
    model=d['selected_A2'];lo=e['summaries']['lomo/'+model];stop=e['summaries']['inner_stop/'+model]
    a0=e['summaries']['inner_stop/A0_frozen'];bench=read('inference_benchmark.csv')
    lines=['# MPPI First-Action Advantage — Phase 3（2026-11-18）',
        '\n## 结论先行',
        f"\n本轮**没有证明 Baseline-relative Advantage 恢复跨地图首动作排序**。按冻结的inner-stop选择规则选中`{model}`：LOMO准确率={fmt(lo['pair_accuracy'],True)}%，Model Gain={fmt(lo['model_gain'])}；inner-stop Regret={fmt(stop['top1_regret'])}，旧A0={fmt(a0['top1_regret'])}。预注册LOMO和inner-stop门槛未通过，按要求停止：**没有新validation、没有在线接入、没有增加网络容量或重训WM/SAC**。",
        '\n不能简单说“所有方法都没信号”：Linear的LOMO Model Gain正区间、KNN的inner-stop排名强，均保留。但这些结果没有同时满足跨图排序、Regret及模型选择门槛，不能事后把更好看的control换成主模型。',
        '\n## 1. 版本、历史时点和数据',
        f"\nparent_commit=`{p['parent_commit']}`；phase2_report_commit=`{p['phase2_report_commit']}`。Phase2正文27f8131是当时生成报告的基线；该报告及代码之后由7d15cdd提交并推送，历史文字不改，本manifest明确两种时点。2026-11-18 GitHub连接器核实main为7d15cdd。",
        f"\nCheckpoint SHA=`{p['checkpoint_sha256']}`，仍为WM200k+RL300k总500k；原535、WM/SAC/Lattice权重、Action Mapping、Reward/Cost、gamma=.995和原生done/成功条件均保留。所有强制阅读的Phase2报告/JSON/CSV与旧训练实现均已全量读取，SHA和结构摘要在experiment_manifest.json。",
        f"\n新目录位于main工作区：`{R}`。Git忽略了旧.pt/.npz，故只读原实验worktree `{PRIOR}` 内的真实二进制，不生成替代旧数据。只打开原fit8图126条H1分支与inner-stop4图64条H1分支，共24个origin；**没有为模型训练/选型打开旧seen-validation分支**。历史报告中的已知验证结论仅作为用户指定固定前提。所有输入文件和源代码SHA均记录并在结束复核。",
        '\n## 2. 严格的same-origin相对目标与输入',
        '\nY取真实首Reward+.995×原冻结确定性Lattice+SAC反馈延续至native done。每个origin根据baseline角色定位唯一真实Baseline，ΔY=Y_i−Y_B，Δa=a_i−a_B；不跨origin/地图计算参考，不用预测score或Oracle作Baseline。原生提前失败与所有候选保留，完整Reward/native done向量重算核验。',
        '\nA0是原冻结F2：current535+绝对物理action2（537），权重完全不变，仅在离线MAE/作图中减去同origin A0 Baseline预测，argmax/排名不变。A1=current535+Δa（537）；主A2=current535+a_B+Δa（539）。没有future latent/obs/Reward/Cost/crash/OOR/隐藏状态作为输入，真实结果只在label/评价侧。',
        '\nA1/A2输出为`[f_theta(x)-f_theta(same context,Δa=0)] × fit_RMS(ΔY)`，Baseline严格为0；不是另一个绝对Return模型。Linear与32→16→1的固定SiLU MLP各训练一套；无Transformer/Ensemble/更大网络。',
        '\n**结构限制**：线性f的状态项、Baseline项与bias被相减消去，最终只有w_delta·Δa。因此A1 Linear和A2 Linear表达类完全相同，本次全部LOMO/inner预测也逐元素相同。这不是“Baseline/context没有价值”的证据，只能在MLP控制中比较其额外信息。',
        '\n## 3. 冻结训练协议及无泄漏LOMO',
        '\nprotocol.json在读取新relative数据前冻结。300固定epoch、seed1、Adam lr=.001，Linear decay=.001、MLP decay=.0001。输入mean/std与label RMS仅取相应fit maps，RMS只缩放不减label均值，保证零参考不被移动。SmoothL1(normalized advantage)+.25×同origin pairwise softplus；pair预测差使用fit尺度归一化，不再raw_difference/10。原abs真实差≤1的Pair排除主排名，tie保留诊断。',
        '\n每图总权重相同、图内每origin总权重相同、origin内候选均分；实际每图2个起点，因此origin全局权重也相同。全fit及各7图fold的scene/origin/candidate权重在training_weights.csv，Pair权重与fit-scale在advantage_pair_diagnostics.csv。没有只训练大动作或删除不利候选。',
        '\n8折均只用7张fit图训练，固定300epoch，不用heldout图、inner-stop或旧seen-validation做早停/归一化。预测只来自未见该地图的那一折，不平均多个fold模型。全fit模型也只用8图，训练结束后一次性评价4张inner-stop；选型按Accuracy DESC、Regret ASC、Gain DESC，稳定tie优先Linear。',
        '\n**A0公平性限制**：旧A0已训练过全部8张fit图，不能提供真实无泄漏的A0 LOMO。遵守“不重训/修改A0、不再训练绝对Return”约束，表中保留它为**seen-fit参考**并显式打标；不把A2对它的LOMO差异当因果比较，也不把泄漏A0纳入LOMO门槛。旧A0还已用inner-stop选过，故本轮inner对比也是开发诊断，不是独立验证。',
        '\nLOMO门槛：选定A2的Accuracy CI下界>.5、Spearman CI下界>0、ModelGain CI下界≥0。inner门槛：点估计Accuracy>.5、Spearman>0、Regret优于A0、Gain≥0、selected-negative不高于A0。两者同时过才允许冻结模型并预注册新的6图/12origin验证。本轮未通过，不继续采图。',
        '\n## 4. 结果：真正held-out地图与开发inner-stop',
        '\n|阶段|模型|Pair Accuracy% [95%CI]|Spearman|Top1 Regret|Model Gain|Selected Negative%|\n|---|---|---:|---:|---:|---:|---:|']
    for stage in ['lomo','inner_stop']:
        for name in ['A0_frozen','A1_linear','A1_mlp','A2_linear','A2_mlp','A2_knn']:
            s=e['summaries'][f'{stage}/{name}'];label=name+(' (seen-fit，非LOMO)' if stage=='lomo' and name=='A0_frozen' else '')
            lines.append(f"|{stage}|{label}|{fmt(s['pair_accuracy'],True)}|{fmt(s['spearman'])}|{fmt(s['top1_regret'])}|{fmt(s['model_gain'])}|{fmt(s['selected_negative'],True)}|")
    lines.extend([
        '\n每个候选/起点/地图层级保留。先起点后图等权，整图2000次Cluster Bootstrap、seed20261118；Pair不当独立地图。8图和4图的区间较宽，单图/单origin不代表总体。Kendall、逐fold Model/Baseline Gain、selected-positive/negative、实际候选选择在对应CSV及evaluation_results.json。',
        '\nA1按同一规则选中Linear；A2选中MLP。A2 inner Accuracy仅略高于Linear，而Regret更差，仍严格沿用已固定的字典序规则；没有看到Regret后再换一个模型来过门槛。KNN仅诊断，不因为inner结果较好就临时改成部署候选。',
        '\n|真正LOMO比较（后者−前者）|Accuracy变化百分点|Regret改善|Gain改善|\n|---|---:|---:|---:|'])
    for key,s in comparisons.items():
        if key.startswith('lomo/'):
            lines.append(f"|{key}|{fmt(s['pair_accuracy_improvement'],True)}|{fmt(s['regret_improvement'])}|{fmt(s['model_gain_improvement'])}|")
    lines.extend([
        '\n## 5. Oracle、动作幅度与简单controls',
        '\nOracle是实际测量代表候选子集中的最大真实ΔY，非完整Gaussian空间或可部署policy。Random沿用旧fixed Gaussian角色，不重新抽有利样本。Baseline=0；逐origin的A0/A1/A2各variant、Random、Oracle和选正/选负都保留在oracle_comparison.csv。',
        '\n|数据|Oracle Gain|至少含>1收益候选的origin比例%|Random Gain|\n|---|---:|---:|---:|'])
    for stage in ['lomo','inner_stop']:
        s=e['oracle'][f'{stage}/Oracle'];random=e['oracle'][f'{stage}/Random'];lines.append(f"|{stage}|{fmt(s['gain'])}|{fmt(s['any_positive'],True)}|{fmt(random['gain'])}|")
    lines.extend([
        '\n开发数据Oracle明显存在正收益，故不能把本轮主模型失败说成“整个candidate space无价值”。但Oracle事后最大化含Baseline，必然非负；既有Phase2新地图Oracle不稳定的固定前提也没有被推翻。本轮无新独立地图，不能宣称潜力已广泛泛化。',
        '\nKNN使用与A2 Linear相同539维当前输入，fit-only标准化后按current535、Baseline2、Δa2分块等权距离，固定k8与距离倒数×训练层级权重；再减去同context零Δa的预测，Baseline仍0。没有试不同k或事后调距离。它在inner很强但LOMO失败，提示邻域/地图覆盖不稳，不能仅凭一个开发split指认MLP容量为唯一问题。',
        '\n动作L2固定层<.01、.01–.05、.05–.1、≥.1，全部样本保留。为可比较的Gain，分层内额外保留Baseline参考；子集Top1是限制候选池的离线描述，不是原在线实际选中动作。缺乏有效Pair的层记N/A，不填0；分层Accuracy/Spearman/Regret/Gain区间见action_magnitude_analysis.csv与evaluation_results.json。',
        '\n## 6. 计算开销（仅离线head，不是完整MPPI）',
        '\nCPU单线程、GPU同设备输入，warmup20/repeats200；CPU/GPU均报告含两次baseline-centered推理的wall时间，GPU另列CUDA kernel时间。参数量、batch1、batch258、tensor峰值显存和CPU/GPU数值差异完整在inference_benchmark.csv。数据传输未包括，tensor显存不是驱动全部进程显存。',
        '\n|模型|参数量|CPU batch258 P50/P95 ms|GPU batch258 wall P50/P95 ms|GPU模型tensor bytes|\n|---|---:|---:|---:|---:|'])
    for name in ['A0_frozen','A1_linear','A1_mlp','A2_linear','A2_mlp','A2_knn']:
        cpu=next(r for r in bench if r['model']==name and r['device']=='cpu' and r['batch']=='258')
        gpu=next((r for r in bench if r['model']==name and r['device']=='cuda' and r['batch']=='258'),None)
        gv=f"{float(gpu['wall_p50_ms']):.3f}/{float(gpu['wall_p95_ms']):.3f}" if gpu else 'N/A'
        lines.append(f"|{name}|{cpu['parameters']}|{float(cpu['wall_p50_ms']):.3f}/{float(cpu['wall_p95_ms']):.3f}|{gv}|{gpu['model_tensor_bytes'] if gpu else 'N/A'}|")
    lines.extend([
        '\nCentered Linear虽有538/540个参数槽，纯线性context/Baseline项不可识别，实际动作表达为2个系数。KNN无神经参数但保存fit数据。小head变快不等于完整规划变快；原完整规划P95 B≈76ms、M1≈104ms、M2≈117ms、M4≈141ms仍为历史基准，本轮没有改变原采样预算或在线时间。',
        '\n## 7. 十一个必须回答的问题与下一步',
        '\n1. **Absolute→relative是否改善？** 没有形成同时通过LOMO与inner-stop的改善；A2 inner Regret明显较差。A0 LOMO看过fit，禁止据其直接证明目标因果优劣。',
        '\n2. **state-value干扰是否被证实？** 未证实。主模型失败不能单独反证state-value假设；本轮同时改变相对目标、局部表示、中心化、归一化/排序loss和小网络，不能把结果唯一归因某一个因素。',
        '\n3. **Δaction是否更易泛化？** 尚无可靠证据。相对Linear有正LOMO Gain，但Accuracy区间仍跨50%；与旧A0的不同结构/目标存在混杂，不能叫单变量结论。',
        '\n4. **Baseline action有额外信息？** A2 vs A1 MLP差异不稳定；Linear中该项数学上必然抵消。不能从Linear相同断言Baseline信息无用。',
        '\n5. **MLP优于Linear？** 没有；LOMO Linear Gain为正，主A2 MLP为负，配对差异上表保留，不扩网络。',
        '\n6. **MLP优于KNN？** 依split而变。KNN inner排名强但LOMO更差，既不支持稳定部署KNN，也不足以唯一归因容量。',
        '\n7. **LOMO是否仍失败？** 主A2未过三项预注册LOMO门槛；Linear有部分积极信号，不把“未过门槛”夸大为“所有低层动作都无用”。',
        '\n8. **主要问题是什么？** 当前可观测瓶颈仍是action-advantage跨地图/context泛化，训练/容量有辅助迹象，不能唯一归因。开发Oracle正收益不支持立即否定candidate空间；本轮没有证据把Dynamics升级为主要原因。',
        '\n9. **Oracle是否稳定正收益？** 本轮fit/inner开发数据有积极区间；跨新地图的稳定性未获得新证据，不能把开发Oracle冒充泛化。',
        '\n10. **是否进入新validation？** 否，开发门槛已失败，新的6图/12origin没有采集，旧seen-validation没有复用来重新选模型。',
        '\n11. **是否具备在线条件？** 否；所有新head保持离线，没有修改MPPI/WM/SAC/Lattice/原配置。',
        '\n唯一下一阶段方向：若继续研究，应预注册增加不同地图/道路交互条件下的origin覆盖，检验当前特征下的局部优势可辨识性；不要继续增加同类网络宽度/深度，也不要因本次结果直接重训WM或重构candidate空间。本轮按门槛停止，不自动新增采集。',
        '\n## 8. 测试、保留和交付',
        '\n原82项在保有历史二进制fixture的旧worktree独立进程全部通过；新增16项在main新目录另一个进程通过，合计98项。首次把两个不同工作区放同一pytest进程触发了原有mpr_mpc导入保护，日志保留；改为独立进程，没有移除保护或放宽任何原断言。测试通过不等于算法有效。',
        '\n新增回归覆盖Baseline Δa/优势精确0、raw差值、same-origin真实target、原gamma/Reward/done、同origin Pair与原tol、场景/origin平衡、fit-only normalization、LOMO无泄漏、Linear/KNN相同输入、无未来输入、A0 SHA/旧inner结果一致、原在线源码不变、seen-validation不选型、Linear中心化消去context。',
        '\n结束复核：Checkpoint/A0 SHA不变，所有受保护在线源码与config逐项SHA不变，全部旧Phase2输入2631项及本轮fit/inner来源文件SHA复核；无tracked在线diff、无commit/push、无新环境回合或正式100场。所有新模型只存新目录，不覆盖旧head或WM/SAC。',
        '\n图均另存PDF：'])
    for name in ['absolute_vs_relative_accuracy','leave_one_map_out','oracle_vs_model_gain','action_delta_accuracy','regret_comparison']:
        lines.append(f'\n![{name}](figures/{name}.png)')
    (R/'MPPI_FIRST_ACTION_ADVANTAGE_REPORT_20261118.md').write_text('\n'.join(lines)+'\n')
    sources=sorted(R.glob('*.py'))+[R/'protocol.json',R/'feature_schema.json']
    patch=[]
    for f in sources:
        result=subprocess.run(['git','diff','--no-index','--','/dev/null',str(f.relative_to(ROOT))],cwd=ROOT,capture_output=True,text=True)
        assert result.returncode==1;patch.append(result.stdout)
    (R/'code_changes.patch').write_text(''.join(patch))
    dump('verification.json',dict(parent_commit=p['parent_commit'],phase2_report_commit=p['phase2_report_commit'],
        checkpoint_sha256=p['checkpoint_sha256'],A0_sha256=p['A0_sha256'],old_inputs_sha_checked=len(old_inventory),
        protected_sources_unchanged=True,default_config_unchanged=True,old_models_unchanged=True,baseline_advantage_exact_zero=True,
        fit_only_normalization=True,same_origin_targets=True,scene_origin_balanced=True,heldout_maps_never_fit=True,
        A0_lomo_disclosed_as_seen_fit=True,A0_inner_accuracy_unchanged=True,linear_A1_A2_predictions_exactly_equal=linear_equal,
        old_seen_validation_not_reselected=True,new_validation_performed=False,new_environment_runs=0,
        relative_models_trained=36,knn_diagnostic_fits=9,absolute_models_retrained=0,
        tests_original82='82 passed in 4.66s',tests_new16='16 passed in 1.60s',tests_total=98,
        tests_in_independent_worktree_processes=True,first_mixed_process_attempt_preserved=True,
        online_changes=False,formal100=False,commit=False,push=False,
        source_sha256={str(f.relative_to(ROOT)):sha(f) for f in sources},
        no_online_value_deployment=True,models_kept_offline=True))
    dump('execution_manifest.json',dict(cwd=str(ROOT),binary_source=str(PRIOR),
        environment={'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1'},
        scripts=['prepare.py','train.py','evaluate.py','benchmark.py','test_advantage.py','report.py'],
        testing='Original82 under prior worktree; new16 under main, --import-mode=importlib, each fresh process.',
        old_test_args=load(PHASE2/'test_command.json'),new_test_args=['python','-m','pytest','-q','--import-mode=importlib',str(R/'test_advantage.py')],
        protocol_sha256=sha(R/'protocol.json'),all_outputs_exclusive_creation=True))
    print('PHASE3 COMPLETE: development gates failed, no new validation;98 tests; original online stack untouched.',flush=True)


if __name__=='__main__':main()
