# AgentBench 仓库任务调研报告（2026-04-13）

> 调研范围：当前 `main` 分支代码与配置文件。
> 重点回答：
> 1. 支持哪些任务；
> 2. 是否有训练/测试（或 dev/std）划分；
> 3. tool API 是任务级固定还是题目级独立。

---

## 1. 总览结论

- 仓库任务配置聚合文件 `configs/tasks/task_assembly.yaml` 当前包含 9 类任务：
  - `webshop`、`dbbench`、`mind2web`、`card_game`、`kg`、`os`、`ltp`、`alfworld`、`avalon`
- 但当前仓库内可见的任务服务端实现目录（`src/server/tasks/`）只有 5 类：
  - `alfworld`、`dbbench`、`knowledgegraph`、`os_interaction`、`webshop`
- 其余任务（如 `mind2web/card_game/ltp/avalon`）在本仓库主要体现为配置项，具体环境行为多依赖外部镜像或外部实现。

---

## 2. 每个任务的详细情况（含 split 与 tools）

> 说明：
> - “划分”列使用任务配置中的命名（如 `dev/std/env_train`）。
> - “Tools”列优先来自 `configs/tasks/*.yaml` 的 `default.parameters.tools`；
>   对 KG，tools 来自 `src/server/tasks/knowledgegraph/const.py`。

| 任务 | 配置项（示例） | 数据划分情况 (题数) | Tools 列表 | 工具粒度判断 | 本仓库实现状态 |
|---|---|---|---|---|---|
| ALFWorld | `configs/tasks/alfworld.yaml` (`alfworld-std`, `alfworld-env_train`) | 有：`new_std` (109题) 与 `train_valid` (3150题) | `take_action` | 任务级固定（在任务默认配置中定义） | 有（`src/server/tasks/alfworld/`） |
| DBBench | `configs/tasks/dbbench.yaml` (`dbbench-std`, `dbbench-env_train`) | 有：`standard.jsonl` (300题) 与 `db_out_new.jsonl` (4803题) | `execute_sql`、`commit_final_answer` | 任务级固定（`default.parameters.tools`） | 有（`src/server/tasks/dbbench/`） |
| Knowledge Graph (KG) | `configs/tasks/kg.yaml` (`kg-std`, `kg-env_train`) | 有：`std.json` (150题) 与 `kg_rl_all.json` (1214题) | `get_relations`、`get_neighbors`、`intersection`、`get_attributes`、`argmax`、`argmin`、`count` | 任务级固定（在 `const.py` 中定义 `TOOLS` 并注入） | 有（`src/server/tasks/knowledgegraph/`） |
| OS Interaction | `configs/tasks/os.yaml` (`os-std`, `os-env_train`) | 有：`os-std` 目录合集 (144题) 与 `training.json` (1000题) | `bash_action`、`finish_action`、`answer_action` | 任务级固定（`default.parameters.tools`） | 有（`src/server/tasks/os_interaction/`） |
| WebShop | `configs/tasks/webshop.yaml` (`webshop-std`, `webshop-env_train`) | 有：配置索引切分 (std: 200题, env_train: 11000题) | `search_action`、`click_action` | 任务级固定（`default.parameters.tools`） | 有（`src/server/tasks/webshop/`） |
| Mind2Web | `configs/tasks/mind2web.yaml` (`m2w-dev`, `m2w-std`) | 有：`data_dev/*.json` (待核实) 与 `data_std/*.json` (待核实) | 配置中未显式声明 tools（多由外部环境/镜像流程决定） | 倾向任务级固定（但需外部实现确认） | 本仓库内未见对应 `src/server/tasks/mind2web/` 实现目录 |
| Card Game | `configs/tasks/card_game.yaml` (`cg-dev`, `cg-std`) | 有：dev/std（由任务名区分；具体样本由外部镜像/环境侧管理） | 配置中未显式声明 tools | 倾向任务级固定（但需外部实现确认） | 本仓库内未见对应 `src/server/tasks/card_game/` 实现目录 |
| LTP | `configs/tasks/ltp.yaml` (`ltp-dev`, `ltp-std`) | 有：`dev.xlsx` 与 `standard.xlsx` | 配置中未显式声明 tools | 倾向任务级固定（但需外部实现确认） | 本仓库内未见对应 `src/server/tasks/ltp/` 实现目录 |
| Avalon | `configs/tasks/avalon.yaml` (`avalon-dev-naive`, `avalon-dev-single`) | 当前可见 dev 划分（`data/avalon/dev.json`），未见 std/env_train | 配置中未显式声明 tools | 倾向任务级固定（但需外部实现确认） | 本仓库内未见对应 `src/server/tasks/avalon/` 实现目录 |

---

## 3. 关于“tool API 是否每题独立”的结论

### 3.1 核心结论

在当前仓库可确认的任务（尤其是已实现的 5 个）中，**tool API 以任务为单位固定定义**，不是“每道题有一套独立工具定义”。

### 3.2 依据

- 在 `dbbench/os/webshop/alfworld` 中，tools 定义于任务配置默认段（`default.parameters.tools`），`std` 与 `env_train` 通常只切数据源/索引。
- 在 KG 中，tools 固定在 `src/server/tasks/knowledgegraph/const.py` 的 `TOOLS`，任务初始化时注入（`super().__init__(tools=TOOLS, ...)`）。
- 任务运行时会按样本变化的是：题目内容、环境状态、可选动作参数/观测，不是替换整套 API Schema。

---

## 4. 各任务具体 Tool API 列表

以下是当前已在仓库中实现的 5 个核心任务的环境交互 API（Tools）列表：

### 4.1 ALFWorld
- **`take_action`**
  - **描述**：Take an action.
  - **参数**：
    - `action` (string): The action you would like to take.
  - **示例与可用命令枚举**：
    ALFWorld 在每一轮（turn）的环境观察后，会返回一个动态的 `admissible_commands`（可用操作列表）。通过 `take_action` 时，必须原样传入这些动词短语。常见有效的操作枚举包括：
    - `go to {receptacle}` (例：`go to diningtable 1`)
    - `take {object} from {receptacle}` (例：`take apple 1 from diningtable 1`)
    - `put {object} in/on {receptacle}` (例：`put apple 1 in fridge 1`)
    - `open {receptacle}` (例：`open fridge 1`)
    - `close {receptacle}` (例：`close fridge 1`)
    - `toggle {object}` (例：`toggle desklamp 1`)
    - `clean {object} with {receptacle}` (例：`clean apple 1 with sinkbasin 1`)
    - `heat {object} with {receptacle}` (例：`heat apple 1 with microwave 1`)
    - `cool {object} with {receptacle}` (例：`cool apple 1 with fridge 1`)
    - `slice {object} with {object}` (例：`slice apple 1 with knife 1`)
    - `examine {object}` (例：`examine apple 1`)
    - **调用 Example**：`{"action": "go to fridge 1"}`

### 4.2 DBBench
- **`execute_sql`**
  - **描述**：Executes a given SQL statement on the database and returns the result.
  - **参数**：
    - `query` (string): The SQL query to be executed.
  - **调用 Example**：`{"query": "SELECT count(*) FROM employees WHERE age > 20;"}`

- **`commit_final_answer`**
  - **描述**：Commits the final answer after all operations are completed.
  - **参数**：
    - `answers` (array of string): The list of final answers to commit.
  - **调用 Example**：`{"answers": ["25"]}`

### 4.3 Knowledge Graph (KG)
> 注：KG 允许传入变量（以 `#` 前缀，如 `#0` 代表上一步的结果）或实体文本。

- **`get_relations`**
  - **描述**：Fetches all relations associated with a given entity or variable...
  - **参数**：`variable` (string)
  - **调用 Example**：`{"variable": "Gas-generator cycle"}`

- **`get_neighbors`**
  - **描述**：Returns all entities connected to a given variable through a specified relation...
  - **参数**：`variable` (string), `relation` (string)
  - **调用 Example**：`{"variable": "#0", "relation": "spaceflight.rocket_engine_cycle.rocket_engines"}`

- **`intersection`**
  - **描述**：Computes the intersection of two sets of entities or variables of the same type.
  - **参数**：`variable1` (string), `variable2` (string)
  - **调用 Example**：`{"variable1": "#1", "variable2": "#2"}`

- **`get_attributes`**
  - **描述**：Retrieves all numerical attributes of a given variable...
  - **参数**：`variable` (string)
  - **调用 Example**：`{"variable": "Mount Everest"}`

- **`argmax`**
  - **描述**：Finds the entity with the maximum value of the specified attribute...
  - **参数**：`variable` (string), `attribute` (string)
  - **调用 Example**：`{"variable": "#3", "attribute": "elevation"}`

- **`argmin`**
  - **描述**：Finds the entity with the minimum value of the specified attribute...
  - **参数**：`variable` (string), `attribute` (string)
  - **调用 Example**：`{"variable": "#3", "attribute": "elevation"}`

- **`count`**
  - **描述**：Counts the number of entities within a given variable.
  - **参数**：`variable` (string)
  - **调用 Example**：`{"variable": "#4"}`

### 4.4 OS Interaction
- **`bash_action`**
  - **描述**：Execute bash code to perform an operation in the Linux environment.
  - **参数**：
    - `script` (string): The bash script to be executed.
  - **调用 Example**：`{"script": "ls -al /var/log"}` 或 `{"script": "cat /etc/os-release"}`

- **`finish_action`**
  - **描述**：Indicate that the task has been finished or need some additional information to be finished.
  - **参数**：
    - `thought` (string): The thought or reason indicating the task is finished.
  - **调用 Example**：`{"thought": "I have created the required directories and verified their permissions."}`

- **`answer_action`**
  - **描述**：Provide the answer to the question.
  - **参数**：
    - `answer` (string): The answer to the question.
  - **调用 Example**：`{"answer": "ubuntu"}` （注意 OS 提示词常要求如果答案为单一数值或词，直接给精确词而不要给整个句子）

### 4.5 WebShop
- **`search_action`**
  - **描述**：Use search functionality with specified keywords.
  - **参数**：
    - `keywords` (string): The keywords to use in the search function.
  - **调用 Example**：`{"keywords": "red nike running shoes size 10"}`

- **`click_action`**
  - **描述**：Click a button or link with a specified value.
  - **参数**：
    - `value` (string): The value to click from the list of available actions.
  - **调用 Example**：`{"value": "Buy Now"}` 或是 `{"value": "B07XY"}` （根据每一轮 WebShop 页面给出的可用 clickable items 决定）

---

## 5. 与“默认运行配置”的关系（补充）

- 虽然任务聚合配置覆盖多任务，默认轻量评测配置仍聚焦 `dbbench-std` 和 `os-std`：
  - `configs/start_task.yaml`
  - `configs/start_task_lite.yaml`
  - `configs/assignments/default.yaml`
  - `configs/assignments/lite.yaml`

这说明：仓库具备多任务配置入口，但默认开箱路径是低资源任务子集。

---

## 6. 证据文件索引

- 任务聚合：`configs/tasks/task_assembly.yaml`
- 任务配置：
  - `configs/tasks/alfworld.yaml`
  - `configs/tasks/dbbench.yaml`
  - `configs/tasks/kg.yaml`
  - `configs/tasks/os.yaml`
  - `configs/tasks/webshop.yaml`
  - `configs/tasks/mind2web.yaml`
  - `configs/tasks/card_game.yaml`
  - `configs/tasks/ltp.yaml`
  - `configs/tasks/avalon.yaml`
- 已实现任务目录：`src/server/tasks/`
- KG 工具常量：`src/server/tasks/knowledgegraph/const.py`
- 仓库说明：`README.md`、`docs/Introduction_en.md`

---

## 7. 如何运行测试（以 ALFWorld 和 WebShop 为例）

1. **环境准备**：
   由于该框架使用 Docker 容器与大模型 API 进行交互，首先确保启动了 Controller 及相应任务环境（例如 `env_train` 划分）：
   ```bash
   docker compose -f extra/docker-compose.yml up -d redis controller alfworld-env_train webshop-env_train
   ```

2. **配置分配策略**：
   在 `configs/assignments/alfworld_webshop_train.yaml` 中配置大模型（如 `gpt-4`、`claude` 等）及对应任务。
   示例：
   ```yaml
   agent:
     # 配置你的 API agents
   tasks:
     - alfworld-env_train
     - webshop-env_train
   ```

3. **启动测试脚本**：
   本地根目录下提供了一个快捷运行脚本 `run.sh`。执行该脚本将启动评测任务并分发至各 Docker Worker：
   ```bash
   ./run.sh
   # 或者手动执行: python -m src.assigner -c configs/assignments/alfworld_webshop_train.yaml
   ```

4. **如何配置子集测试（打乱并抽样跑部分题目）**：
   如果你觉得训练集太大，可以修改对应任务配置（如 `configs/tasks/alfworld.yaml` 或 `configs/tasks/webshop.yaml`）里的 `parameters`，增加 `shuffle_seed`、`start`、`end` 或 `sample_size` 字段来固定打乱后抽取指定数量的题。

   **ALFWorld 示例** (`configs/tasks/alfworld.yaml`)：
   ```yaml
   alfworld-env_train:
     parameters:
       name: alfworld-env_train
       split: "train_valid"
       shuffle_seed: 42     # 固定随机种子
       start: 0             # 抽取切出前 100 个
       end: 100
   ```
   
   **WebShop 示例** (`configs/tasks/webshop.yaml`)：
   ```yaml
   webshop-env_train:
     parameters:
       name: webshop-env_train
       start: 1000          # 源数据的起始切片
       end: 12000           # 源数据的结束切片
       shuffle_seed: 42     # 固定随机种子
       sample_size: 100     # 从中等概率抽取 100 题
   ```

   **LLM API 配置文件设置**：
   测试必须使用一个 assignment yaml 文件，里面需要写明 agent (你的 API) 和被评测的 task。
   我们在 `configs/assignments/alfworld_webshop_train.yaml` 中将上述子集与 Agent 绑定即可运行。

5. **查看测试结果**：
   运行结束后，测试结果、Token 消耗、成功率以及每个容器内运行的具体交互 Trace（JSONL格式）可以在 `outputs/` 目录找到。

---

## 8. 可执行的下一步（可选）

如需，我可以继续补一版“机器可读清单”（JSON/YAML），字段包含：`task_name`、`implemented_in_repo`、`splits`、`tools`、`data_source`、`confidence`，便于你直接喂给训练/评测 pipeline。