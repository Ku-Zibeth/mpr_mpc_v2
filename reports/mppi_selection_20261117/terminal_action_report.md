# terminal action report

本节是同一真实闭环实验的摘录，不是额外验证。完整协议和限制见仓库根目录MPPI_CLOSED_LOOP_SELECTION_REPORT_20261117.md。

## Terminal Action和Mean/Elite消融（固定2轮）

|Terminal / Selection|Success%|Crash%|OOR%|Route%|Episode Return|实际折扣Return|相对B折扣Return [95% CI]|
|---|---:|---:|---:|---:|---:|---:|---:|
|Candidate / Mean /2|50.0|0.0|50.0|85.0|319.26|141.71|-3.019 [-4.890, -1.395]|
|Policy prior / Mean /2|37.5|0.0|62.5|79.5|299.48|136.75|-7.974 [-15.287, -3.865]|
|Candidate / Final Elite /2|37.5|0.0|62.5|79.7|287.29|137.00|-7.730 [-15.877, -2.388]|
|Policy prior / Final Elite /2|50.0|0.0|50.0|85.3|312.42|140.80|-3.925 [-7.214, -1.033]|

|单变量配对比较|Success变化百分点 [CI]|OOR变化百分点 [CI]|折扣Return变化 [CI]|
|---|---:|---:|---:|
|Policy prior−Candidate（Mean）|-12.5 [-37.5, 0.0]|12.5 [0.0, 37.5]|-4.954 [-13.937, 0.788]|
|Elite−Mean（Candidate）|-12.5 [-37.5, 0.0]|12.5 [0.0, 37.5]|-4.711 [-14.664, 2.829]|
|Elite−Mean（Policy prior）|12.5 [0.0, 37.5]|-12.5 [-37.5, 0.0]|4.049 [-3.360, 13.361]|
|Policy prior−Candidate（Elite）|12.5 [0.0, 37.5]|-12.5 [-37.5, 0.0]|3.805 [-3.390, 13.929]|

单独采用policy_prior或Final Elite均为37.5%成功、62.5%OOR，差于B及Candidate+Mean的50%/50%；组合回到50%，但折扣Return仍低3.92。不能把组合比某个较差消融更好解释为超过SAC；没有启用未验证Cost或加入新规则补救。

![真实配对收益](reports/mppi_selection_20261117/figures/paired_return.png)



结论：开发集未获相对SAC的可重复正收益；验证保持封存，默认配置及checkpoint未变。
