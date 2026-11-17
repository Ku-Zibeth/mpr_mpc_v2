import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];R=Path(__file__).parent

def load(name):return json.loads((R/(name+'.json')).read_text())
def n(x,d=3):return 'NA' if x is None else f'{x:.{d}f}'
def ci(r,key,percent=False):
 factor=100 if percent else 1
 def v(x):return 'NA' if x is None else n(x*factor,1 if percent else 3)
 return f"{v(r.get(key))} [{v(r.get(key+'_ci95_low'))}, {v(r.get(key+'_ci95_high'))}]"

p=load('protocol');audit=load('data_audit');cov=load('data_coverage');manifest=load('dataset_manifest');ev=load('ranking_evaluation');cf=load('counterfactual_results');search=load('search_diagnostics');verification=load('verification');primary=p['training']['primary'];passed=ev['validation']['passed'];lines=[
'# 同起点候选排序学习：数据审计与独立验证',
'',f"基准 `{p['base_commit']}`，开始时工作区干净。只增加离线评分器/实验与3项针对性测试；原WM/Q/SAC/Lattice、535维输入、动作映射、Reward/gamma/done、Gaussian/AR(1)、64/8/4及所有std/rho/温度均未修改。原checkpoint SHA256为 `{p['checkpoint_sha256']}`。没有commit/push/reset，没有训练原WM/SAC。",
'', '## 数据和标签审计',
'',f"上一轮报告中的“72条分支”应更正为：72个计划任务，**69条真实完成轨迹、3个不可达起点任务**。60566:150的原SAC在预定起点前结束；没有补造未来状态，也没有删除这些不可达记录。原数据包含12个场景、23个可达起点；8个拟合场景16个起点，48个同起点Pair。2676个拟合窗口和1017个内部早停窗口不是同等数量的独立监督样本。历史文件未覆盖，本报告修正计数。",
'',f"所有69条轨迹、3693个已保存窗口均重新核验原始tdmpc2_reward和原gamma=.995的RTG；targets严格等于rtg[start+10]。重新用原冻结WM编码/推演，预测latent和真实后继latent最大重建误差均为{audit['reconstruction_max_error']['predicted_latent']}/{audit['reconstruction_max_error']['actual_latent']}。需要与原GPU batch大小一致；首次不同batch的浮点差异不能当标签错位。原动作前缀与采集前固定候选一致，同起点观测/车辆/交通/可记录RNG/候选完全匹配；这不是完整隐藏Bullet快照。",
'',f"本轮预先固定近似相等范围为abs(真实总折扣收益差)<=1.0。旧拟合Pair中{cov['old_coverage']['effective_fit_pairs']}/48对超过该范围；全部69对中{audit['effective_pairs']}对有效。实际收益差逐对保存在data_audit.json；旧分支含{cov['old_coverage']['crash_branches']}条碰撞、{cov['old_coverage']['offroad_branches']}条越界，不是全安全数据。动作扰动与正负收益差、各场景有效Pair数见data_coverage.json。",
'',f"相邻滑动窗口H10/stride5有50%前缀重叠；RTG lag-1相关中位数{n(cov['old_coverage']['lag1_rtg_correlation'].get('p50'))}，完全相同观测+H动作窗口重复{cov['old_coverage']['exact_duplicate_windows']}个。后续窗口还含执行中的反馈SAC动作，不等于在该起点事先生成的MPPI候选。本轮拟合只使用每个真实分支的最初预定起点，保留旧原始窗口用于审计，不提高滑窗密度。",
'', '旧损失分离的证据与解释：原训练batch均值与精确场景加权早停损失的计算并不完全相同；更关键的是2676个绝对回归窗口只配48个同起点差分约束，独立地图早停误差随拟合降低而升高。这支持场景层面拟合和泛化不足，但不是标签错位证据。新训练每步用同一完整分组batch、训练和早停用相同目标，避免混用损失统计。',
'', '## 新的候选导向数据',
'', '新增场景和所有候选种类在protocol.json预先划定：12个拟合、4个内部早停、6个本轮新验证场景，范围62000–64999；保留旧8拟合/4早停的原始角色。之前使用过的6个验证场景、旧20场和正式测试均未用于本轮拟合、早停或新验证。验证采集可并行，但其标签不进入拟合代码；五个模型和各自epoch保存后才执行一次新验证，没有在验证后重训、反转分数或挑选输入版本。',
'', '每场景固定step50/150起点。四条候选：SAC baseline、初始64个Gaussian中的固定第一条、原当前Q-MPPI最终加权Mean、统一共同Q参考复评的256条真实采样Best Elite。候选在实际收益产生前生成，保持原采样参数。分别在新进程严格匹配起点，执行前10步，再由原SAC直到原生终止。保存真实reward/cost/done/terminated/truncated、实际观察、当前/预测/真实latent及全部实际控制。发生H前终止的真实总收益保留，不编造H末端真实latent；绝对末端Value损失只用实际存在的末端。',
'', '|集合|独立场景|可达起点|原始候选分支|去重后Pair|有效Pair|', '|---|---:|---:|---:|---:|---:|']
for split in ['fit','inner_stop','validation']:
 x=cov[split];lines.append(f"|{split}|{x['scenes']}|{x['origins']}|{x['raw_branches']}|{x['pairs']}|{x['effective_pairs']}|")
lines += ['', '所有实际回报较好/较差、近似相等、碰撞、越界分支均保留；没有按实际Return事后筛选。精确相同H控制只合并训练权重，并核对真实回报一致；仍保留原文件和重复记录。按场景、起点和候选平衡；排序loss排除近似相等对后，会按每个起点的有效Pair数重新平衡。CI重采样整张地图，不把Pair或窗口视作独立地图。独立起点的速度、曲率、车流分布及逐场景Pair数量见data_coverage.json。',
'', '## 评分器与训练目标',
'', '新增planning/candidate_value.py。共享标量S(z0,imagined_zH,A[:10])保证偏好传递性，训练预测差S_i−S_j，标签为两个真实H步前缀加同一SAC延续的总折扣回报差。主输入不包含真实未来观察/奖励/Cost/实际末端latent/隐藏状态，也不包含未执行a_H。z0和动作是规划时可得量。baseline是共同常数，不额外把baseline动作送入网络；相对SAC分数通过共享S(A)−S(A_SAC)得到。',
'', '主损失为Logistic Pairwise Ranking + 0.25×SmoothL1 Difference。Difference以仅拟合地图上平衡Pair RMS归一化，方向损失优先；近似相等对只保留差分回归。三个诊断输入：A预测zH、B当前z0+预测zH、C再加H步动作。约74k参数、两层SiLU，无Transformer。所有输入均值/方差只来自拟合数据；初始/末端latent相关性和动作尺度单独记录，不能用验证数据标准化。',
'', '消融：A重新拟合绝对V回归；B同架构绝对V+更充分的同起点总收益差SmoothL1；C共享排序头的三种输入。旧冻结V_SAC另外作为对照，未覆盖其权重。新绝对V分数仍为预测prefix+gamma^H×V；共享排序S直接估计候选相对优劣，不再混用未执行a_H的Q。',
'', '|模型|参数量|训练epoch|内部早停最佳epoch|最佳epoch训练loss|早停loss|训练排序率|早停排序率|', '|---|---:|---:|---:|---:|---:|---:|---:|']
for m in manifest['models']:
 tr=m['train'];lines.append(f"|{m['model']}|{m['parameters']}|{m['epochs']}|{m['best_epoch']}|{n(tr['train_loss'])}|{n(tr['inner_stop_loss'])}|{n(tr['train_pair_accuracy'])}|{n(tr['stop_pair_accuracy'])}|")
lines += ['', f"Adam lr=3e-4，最多150步完整分组batch、patience20、固定seed1。差分归一化尺度{n(manifest['difference_scale'])}；rank_context_actions在看新验证前即为主模型。原始训练曲线/模型hash在*_training.json与models/。没有使用验证排名选择五个模型中的赢家。不同目标loss的绝对大小不可横向比较。", '', '五种模型均在第1步取得最低内部早停loss。主模型第21步拟合排序率已到81.7%，但早停仅40.7%；保存的是第1步权重，而不是后期过拟合权重。由完整同目标统计仍出现分离可见，上一轮的问题不只是batch损失口径不同；加入差分/排序损失和上下文没有自动解决地图泛化。', '', '## 一次新独立场景排序验证', '', '以下为场景聚类bootstrap2000次、seed1的95%CI；未做多重比较校正。主要排序准确率排除abs真实差<=1.0；all_pair_accuracy另含近似相等对。Spearman/Kendall、Top1 Regret、收益差MAE均按同起点候选池计算。共享S的跨起点偏置不可由差分监督识别，因此不报告其无意义的绝对Return MAE。', '', '|评分器|Pair准确率% [CI]|Baseline/Mean准确率% [CI]|Spearman [CI]|Kendall [CI]|Top1 Regret [CI]|收益差MAE [CI]|', '|---|---:|---:|---:|---:|---:|---:|']
for name in ['original_Q','previous_V_SAC',*p['models']]:
 r=ev['results']['validation'][name]['summary'];lines.append(f"|{name}|{ci(r,'pair_accuracy',True)}|{ci(r,'baseline_mean_accuracy',True)}|{ci(r,'spearman')}|{ci(r,'kendall')}|{ci(r,'top1_regret')}|{ci(r,'difference_mae')}|")
lines += ['', '高预测增益但真实下降定义为相对baseline预测增益>1.0、真实增益<−1.0；条件比例、分母和场景CI均单独保存。真实收益差区间[0,1),[1,5),[5,20),[20,∞)、动作RMSE区间[0,.02),[.02,.05),[.05,.1),[.1,∞)，以及道路类型/曲率/车流的分层结果和逐地图结果见ranking_evaluation.json。小样本子组的CI不代表稳定泛化。原Q高预测增益候选中11/23条实际下降（6地图），旧V为7/11（4地图），主新头为4/4（仅2地图）；主头条件bootstrap区间退化为[1,1]只是有限条件样本全部为坏例，不能解释为总体概率已精确确定。',
'',f"独立启用门槛结果：**{passed}**。主模型须Pair准确率下界>50%、Spearman下界>0，并在同一起点对原Q和旧V均显示Regret改善CI下界>0；Baseline/Mean排序与错误选择不能恶化。没有因某个次要消融点估计较好就换主模型。",
'', '## 同起点真实反事实结果',
'', '下面是新独立地图中已经实际执行的四种reward-blind候选，以及冻结评分器从相同候选池中选取分支的结果。**这是H10干预+原SAC延续，不是新评分MPPI全程重规划闭环。** 各分支原生终止，提前失败保留；Episode Return和Safety Cost从干预起点累计。采集期间共享GPU，记录的原Q规划耗时不能用于声称新评分提高实时性。',
'', '|方法|Success%|Crash%|OOR%|Route%|Episode Return|实际折扣Return|平均Safety Cost|实际相对B增益|', '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
for name in ['candidate_baseline','candidate_gaussian','candidate_mean','candidate_best_elite','pool_selected_original_Q','pool_selected_previous_V_SAC','pool_selected_rank_context_actions']:
 r=cf['summary'][name];lines.append(f"|{name}|{n(r['success']*100,1)}|{n(r['crash']*100,1)}|{n(r['out_of_road']*100,1)}|{n(r['route_completion']*100,1)}|{n(r['episode_return'],2)}|{n(r['actual_discounted_return'],2)}|{n(r['average_safety_cost'])}|{n(r['actual_return_gain'])}|")
lines += ['', '完整逐起点真实/预测增益、事故差、路线和各项CI在counterfactual_results.json。没有把从已执行Q候选池中选得更好解释为新评分器优化后的Mean已获真实收益，更没有当作正式Success提升。若排序不通过，本轮止于这些实际分支，不做新评分完整闭环或100场。',
'', '## 搜索一致性、耗时及Cost',
'', f"原Top-K/Weighted Mean/Std实现未改。缓存258候选按新头GPU批量评分；平移到相对baseline后的Top8集合一致性、拆batch稳定性、每轮ESS/熵与Mean/Std检查见search_diagnostics.json，所有更新物理范围/finite/padding检查结果：{search['all_updates_valid']}。这里只重评原Q候选池，不冒充新头的在线迭代。实际采集还核验selected_action==selected_sequence[0]。", '', '|起点|新头258批GPU P50(ms)|旧V258批GPU P50(ms)|Top8拆batch Jaccard|分数拆batch最大误差|', '|---|---:|---:|---:|---:|']
for x in search['rows']:
 lines.append(f"|{x['scenario']}:{x['before']}|{n(x['times']['relative_score']['cuda_p50_ms'])}|{n(x['times']['previous_terminal_V']['cuda_p50_ms'])}|{n(x['top8_batch_jaccard'])}|{n(x['batch_score_max_error'],6)}|")
lines += ['', '这些是采集结束后的头部微基准，不含WM/Lattice/SAC推演、不是完整规划延迟；模型未接在线，不能给出新头全规划耗时改善结论。正式100场未运行。',
'', 'Cost保持weight=0，不改原Validator阈值、不训练Cost、不引入TTC/几何过滤/Recovery/Emergency/其他规划模式。上一轮低Cost MAE不是有效候选排序证据，本轮没有强行启用。',
'', '## 结论与后续',
'', ('排序门槛通过，但尚不能宣称在线优于B；需要新头优化轨迹的开发反事实和另外未使用地图的独立闭环证据。' if passed else '**未满足在线启用条件。保持旧在线输出，不运行更大规模或正式100场实验，不宣称优于B。**'),
'', '证据应分开解释：标签与latent已对齐，没有发现H偏移。拟合20张地图后主头仍表现为训练排序上升、内部早停/新地图排序失败，支持数据覆盖与泛化问题；新增早停32条分支全部成功，而新增拟合/新验证分别有11/16条越界，覆盖分布仍不平衡，但没有按真实结果事后重选样本。terminal-only/context/context+actions的新地图排序率分别为41.9%/42.2%/37.8%，并没有独立证据证明增加动作上下文有效。新验证预测末端与真实编码L2误差中位2.20，而候选间预测末端L2差中位1.68，提示rollout误差可能遮盖排序信号；L2大小本身不能证明Dynamics因果失效。下一步应先增加预先划定的独立起点和搜索候选覆盖，另以完全隔离的真实末端编码对照判断：若真实latent评分可靠但预测latent失败，再聚焦Dynamics；若两者都失败，先查状态条件/表示与标签覆盖，而非直接重训WM。',
'', '本轮不修改WM表示或Dynamics权重，也不在这6张已使用验证地图上继续挑loss、网络或翻转分数。没有满足在线启用前提，因此未接入新在线评分分支；这项集成按任务要求暂缓。采集结束后的状态JSON读取竞争只影响耗时检查调度，已修复原子写入/重读，并保留已训练模型及唯一验证结果继续完成；没有因此重训或覆盖结果。',
'', f"相关测试：**{verification['tests']}**，全部原47项保留，新3项覆盖native H标签/提前终止、共享排序传递性/平移不变/近似相等/权重、在线可得输入与a_H排除。原checkpoint/旧Value/旧数据hash保持，旧测试未修改。完整实验清单、数据SHA、模型、训练曲线、排序/反事实JSON、verification及code_changes.patch保存在reports/candidate_ranking_20261117。"]
(ROOT/'MPPI_RELATIVE_RANKING_REPORT_20261117.md').write_text('\n'.join(lines)+'\n')
