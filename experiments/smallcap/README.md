# 冻结的小盘股GPU实验

`willr_fixed_20261104/` 保存2026-11-04从零重训使用的最新脚本、plan.json和完整上游源码快照，包含neutral-zero-v1与willr-sign-v1。模型生成因子公式；每组选择一个公式用于整个20股股票池。

## 数据

本仓库不分发市场数据。若要复现既有实验，请把原 smallcap_gpu_experiment/data 中的universe.json、各股JSON和必要元数据复制到本目录的 `data/`。experiment.py从这个位置只读加载。

`prepare_data.py` 保留原始数据准备逻辑：新浪当前总市值快照筛选20–100亿元、非ST主板股票，再按市值选择20只；腾讯日线后复权及不复权数据截至2026-10-23。脚本没有随发布运行。当前重跑会取得新的市值/复权版本，可能与截止日价格检查不匹配，不等价于恢复原冻结数据；请勿把新的数据称为原实验。

## 环境

历史训练环境为Linux/WSL、Python3.13、PyTorch2.8.0+cu128、RTX4060。runner要求CUDA，公式评估在CPU；control.py使用fcntl和/proc，仅支持Linux/WSL。

建立独立环境，安装兼容的CUDA PyTorch和requirements-experiment.txt。该源码快照不要求安装MetaTrader5或连接券商。

```bash
cd experiments/smallcap/willr_fixed_20261104
python3.13 -m venv .venv
.venv/bin/python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
.venv/bin/python -m pip install -r requirements-experiment.txt
# 数据准备完成后，由使用者显式启动；本次发布没有启动新训练。
.venv/bin/python -B control.py start
.venv/bin/python -B control.py status
# 按需暂停，当前步结束后保存断点
.venv/bin/python -B control.py pause
```

队列先小后大，均为9000步、batch192、种子20261025；每200步保存完整状态，1000/3000/6000/9000步保存候选。全部完成后统一执行2025选公式与2026回看。数据、源码和配置指纹不符时拒绝恢复。control.py使用当前解释器，launch_manifest.json为这次可发布文件生成的源码校验清单；原机器运行清单和断点没有发布，不应与新目录混用。

## 结果解释

2022–2024训练，2025选择公式，2026截至10月23日回看。仅一个种子；股票池有历史选择偏差。分数仓位、单向0.1%摩擦，另测0.2%；不完整覆盖整手、最低佣金、历史ST、精确涨跌停、分红税和实际盘口。2026已被反复查看，不能再称为新盲测。

本次只同步已完成的代码，未重新训练或操作账户。旧缺陷版本的收益与旧QMT对账不能代替最新修复版验证。
