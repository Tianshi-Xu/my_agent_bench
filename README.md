## Install
```
cd AgentBench
conda create -n agent-bench python=3.9
conda activate agent-bench
pip install -r requirements.txt
```
## Run
#### 1.配置
- 首先需要确认configs文件夹下的三种配置 (以alfworld为例)
  - agents/api_agents.yaml: 确认url是正确的，sglang服务的启动可以参考taubench的start_sglang.py，注意端口号对齐。
  - tasks/alfworld.yaml: alfworld-std和alfworld-env_train分别代表了测试集和训练集，测试的时候只需要跑测试集。`enabled`字段表示是否开启harness，后续的h2/h3等字段表示启动对应层级的harness，做消融实验时，每次只去掉其中一个层级，并保持其他层级打开。
  - assignments/alfworld_train.yaml: 确认assignments是正确的, concurrency可以设置为16。`output`参数表明了结果放到哪个文件夹，可以自由配置整理。
#### 2.运行
- 运行docker，任务是在docker内评测的:
  - ```docker compose -f extra/docker-compose.yml up -d --force-recreate redis controller alfworld-env_train alfworld-std```
  - 说明：redis controller必须启动，后面需要启动的容器取决于跑什么任务，例如上面是跑的alfworld的训练或测试任务。**注意，每次修改配置，都需要重新运行docker启动命令**。
  - 注意：webshop任务启动时间很长，即便docker显示启动了，还需等待数分钟才能运行任务，否则会提示找不到webshop容器，此时等待片刻再试一次就好。
- 运行任务
  - ```python -m src.assigner --config configs/assignments/alfworld_train.yaml```