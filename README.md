# 曜衡智控：燃气智能检测 Agent

面向燃气业务的本地智能检测平台，包含每日全量巡检、智能计量、智能设备、安全作业和智能问答五个模块。后端使用 FastAPI，前端使用 React + TypeScript + Vite，业务数据主要存储在 DuckDB、Parquet 和 SQLite 中。

## 快速开始

### 环境要求

- Python 3.12+
- uv
- Node.js 与 npm

安装 Python 和前端依赖：

```powershell
uv sync
npm --prefix frontend install
```

### 项目结构

后端采用标准 `src` 布局，运行代码与数据、脚本、测试分离：

```text
src/intelligent_detection_agent/  Python 后端包
├─ conversation_agent/            多轮问答与 Skills
├─ rag/                           技术文档检索
└─ safety_operations/             安全作业代码
scripts/                          离线构建、诊断和校验入口
tests/                            后端测试
frontend/                         React 前端
dataset/ database/ models/        本地数据与模型
reports/ output/                  运行产物
```

复制 `.env.example` 为 `.env`，按需填写：

- `DEEPSEEK_API_KEY`、`LLM_BASE_URL`、`LLM_MODEL`：通用大模型配置。
- `CHAT_LLM_*`：智能问答专用模型；未填写时沿用通用配置。
- `ARK_API_KEY`、`ARK_MODEL_ID`：安全作业视频复核。
- `SAFETY_AGENT_TOKEN`：安全作业内部通知鉴权令牌。
- `AUTH_COOKIE_SECURE`：本地 HTTP 使用 `false`，生产 HTTPS 使用 `true`。

### 通过 SSH 使用服务器上的 vLLM 微调模型

在本机 PowerShell 建立隧道（替换 SSH 用户、地址和端口）：

```powershell
& "$env:WINDIR\System32\OpenSSH\ssh.exe" -N `
  -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 `
  -L 127.0.0.1:18000:127.0.0.1:8000 `
  -p 2222 用户名@服务器地址
```

SSH 登录环境必须能访问 vLLM 的 `127.0.0.1:8000`；WSL 部署时，确认 SSH 登录到该 WSL 环境。
输入密码后保持窗口打开，在另一个本机 PowerShell 验证：

```powershell
Invoke-RestMethod http://127.0.0.1:18000/v1/models
```

确认模型列表包含 `agent-sft` 后，在 `.env` 中设置：

```dotenv
CHAT_LLM_PROVIDER=openai-compatible
CHAT_LLM_BASE_URL=http://127.0.0.1:18000/v1
CHAT_LLM_API_KEY=local
CHAT_LLM_MODEL=agent-sft
CHAT_LLM_CONTEXT_WINDOW=20000
CHAT_LLM_CHAT_TEMPLATE_KWARGS='{"enable_thinking":false}'
CHAT_LLM_STREAM_USAGE=true
```

`local` 是无鉴权 vLLM 的客户端占位 Key；服务器配置鉴权时填写真实 Key。
`CHAT_LLM_CHAT_TEMPLATE_KWARGS` 必须是 JSON 对象，通过 `extra_body.chat_template_kwargs` 发送；留空不发送，DeepSeek 分支忽略它。`.env` 中保留外层单引号，避免 `uv --env-file` 移除 JSON 内部双引号。
`CHAT_LLM_THINKING` 只控制 DeepSeek，关闭 Qwen 思考需使用上面的模板参数，参见 [vLLM Qwen 文档](https://docs.vllm.ai/projects/recipes/en/latest/Qwen/Qwen3.5.html)。
服务端还需启用 `--enable-auto-tool-choice --tool-call-parser qwen3_coder` 才能解析工具调用。
`CHAT_LLM_STREAM_USAGE=true` 为兼容接口请求 `stream_options.include_usage`，供评测采集真实 Token；仅接受 `true/false`，留空保持 SDK 默认行为，DeepSeek 分支忽略。服务器未返回用量时仍记录缺失，不估算补数。

上下文配置用于本地自动压缩阈值，不会扩大服务器容量。服务器 `max_model_len=20000` 限制输入与输出的总长度；超长请求会报错，不能沿用教师模型的 1M 配置。
工具在本机执行，服务器负责生成。首次接入验证文本、流式响应和工具调用；模型列表成功不代表多模态链路已验证。

评测前显式填写 `EVAL_JUDGE_PROVIDER/API_KEY/BASE_URL/MODEL` 为教师配置（例如 `deepseek`、教师 Key、`https://api.deepseek.com`、`deepseek-flash`），避免留空后随 Agent 切换成学生模型。密钥只保留在本地 `.env`。
修改配置后重启后端并新建会话；已有进程会缓存模型配置，终端中已有的同名环境变量也会优先于 `.env`。

先运行一个只读问题，验证包括教师 Judge 和 SFT 筛选在内的链路：

```powershell
uv run --env-file .env python -m intelligent_detection_agent.evaluation `
  --suite agent --case query_02_user_profile --repeat 1 --concurrency 1 --distill
```

未达到筛选阈值时 `sft.jsonl` 为空属正常结果，应检查 `agent_cases.jsonl` 的评分和导出原因。
隧道窗口关闭或网络断开后，重新运行 SSH 命令并再次检查模型列表；不会自动切回教师模型。
若 `ssh` 命令找不到，使用上面的完整路径；32 位 PowerShell 访问不到该路径时改用 `$env:WINDIR\Sysnative\OpenSSH\ssh.exe`。

### 开发模式

一个终端同时启动 Vite 和 FastAPI：

```powershell
.\start_api.ps1
```

该命令会先确认本地 Qdrant 已就绪，再启动后端和前端。Qdrant 默认位于
`D:\Qdrant`；其他位置可使用 `-QdrantDir` 指定。

- 前端：`http://127.0.0.1:5173/`
- 后端 API：`http://127.0.0.1:8000/`
- API 文档：`http://127.0.0.1:8000/docs`

Vite 会将业务接口代理到 FastAPI。按 `Ctrl+C` 会同时关闭前后端。

### 生产模式

```powershell
npm --prefix frontend run build
.\start_api.ps1 -ApiOnly
```

构建结果位于 `frontend/dist`，由 FastAPI 直接提供，访问 `http://127.0.0.1:8000/`。

### 登录

平台必须登录后使用。首次启动会幂等创建演示账号：

```text
用户名：admin
密码：123456
```

登录页的“填入演示账号”只填充输入框，不会自动提交。密码使用 Scrypt 摘要保存，七天会话使用 HttpOnly、SameSite=Lax Cookie；退出登录会立即使服务端会话失效。

## 数据与存储

| 文件 | 用途 |
| --- | --- |
| `database/gas_ai_input.duckdb` | 用户、表具、检定、SCADA 和设备输入数据，只读使用 |
| `database/gas_ai_results.duckdb` | 智能计量、智能设备诊断结果、智能解读缓存和业务工单 |
| `database/user_data.db` | 用户、登录会话、对话历史、产物和 LangGraph checkpoint |
| `safety_operations/data/security.db` | 安全事件、PPE 复核、证据、通知发件箱和处置审计 |
| `dataset/telemetry` | 按日期分区的 SCADA Parquet 数据 |
| `dataset/vibration` | 按日期分区的三轴振动 Parquet 数据 |

### 输入数据库结构

`database/gas_ai_input.duckdb` 包含以下逻辑 schema：

1. `asset`：用户与表具档案。
2. `inspection`：流量计检定、检定点和维修记录。
3. `telemetry`：SCADA 时序观测和文件导入日志。
4. `equipment`：振动传感器与企业流量计映射。
5. `vibration`：三轴振动窗口、五阶段健康标签和退化轨迹。

`telemetry.scada_observation` 提供统一 SCADA 查询入口；`vibration.acceleration_window` 和 `vibration.daily_health` 分别提供振动窗口与每日健康状态。振动退化数据由工程师三分类样本派生，标记为 `is_synthetic=true`，不能作为现场真实寿命记录。

### 当前数据规模

- 计量用户与表具档案：535 户。
- 检定记录：3,044 条；检定流量/误差点：9,570 条；维修记录：621 条。
- SCADA 时间范围：2024-12-25 至 2025-01-12，共 19 天。
- SCADA 标准化分管路记录：20,794,808 条。
- 成功读取源文件：14,554 个；未识别管路字段：95 个。
- 三类计量输入均可关联用户：393 户。
- 智能设备企业：715 户；振动窗口：13,585 条，每户每天一条。
- 每个振动窗口包含后盖 X/Y/Z 三轴，每轴 500 点，采样频率 1 kHz。

### 从原始文件重建数据

仅在需要重建数据库时设置源数据目录：

```powershell
$env:GAS_SOURCE_ROOT='D:\path\to\gas-source-data'
$env:GAS_VIBRATION_SOURCE_ROOT='D:\path\to\vibration-source-data'

uv run python .\scripts\build_database.py --mode master
uv run python .\scripts\build_database.py --mode scada --overwrite
uv run python .\scripts\build_vibration_database.py --overwrite
uv run python .\scripts\validate_vibration_database.py
```

## 功能模块

### 每日全量巡检

按日期汇总企业级计量和设备诊断结果，展示异常企业、问题模块、风险等级和高频问题。相同日期使用单任务锁避免开发模式重复计算；已生成报告可从 `reports/daily_precision_overview_v3` 直接读取。

### 智能计量

`src/intelligent_detection_agent/smart_metering.py` 的主要流程：

1. 读取用户、检定和 SCADA 数据。
2. 执行五分钟重采样、异常值处理、缺失填充和完整度评价。
3. 使用 CWT-Inception-SimAM 模型识别用气状态。
4. 交叉验证模型状态和远传计量状态，识别“走气未走字”。
5. 检查压力、温度、双管流量不平衡和传感器异常。
6. 使用历史同时间点中位数和 MAD 建立正常用气基线。
7. 定位实际流量显著低于基线的连续异常区间。
8. 根据检定误差曲线和基线缺口估算补气量。
9. 分析表具小流、正常和超量程运行占比。
10. 生成风险评分和核查工单。

单用户命令行诊断：

```powershell
uv run python .\scripts\metering_cli.py `
  --user-id 2267475 `
  --date 2025-01-12 `
  --output '.\reports\diagnosis_2267475_2025-01-12.json'
```

添加 `--no-model` 可跳过深度模型，只执行规则、基线和量程分析。

### 智能设备

`src/intelligent_detection_agent/smart_equipment.py` 使用三轴振动执行多任务健康诊断：

1. 按 80/120/160 工况执行训练集统计归一化。
2. 通过 3/5/9/17 四尺度数学形态学残差提取冲击和包络特征。
3. 使用尺度注意力和三轴门控融合不同尺度与方向的响应。
4. 联合使用主分类头、CORAL 有序辅助头和健康指数回归头建模 H0-H4。
5. 根据健康指数斜率、最大单日下降和突变前斜率识别退化或恢复趋势。
6. 结果写入 `equipment.health_diagnosis` 和 `equipment.health_trend`。

阶段标签包括 H0 健康稳定、H1 轻微衰减、H2 中度衰减、H3 重度衰减和 H4 故障异常。

```powershell
# 训练
uv run python .\scripts\equipment_cli.py train --epochs 20 --batch-size 256

# 单日诊断
uv run python .\scripts\equipment_cli.py diagnose `
  --user-id 1071586391 --date 2025-01-12

# 趋势诊断
uv run python .\scripts\equipment_cli.py trend `
  --user-id 1071586391 --start 2024-12-25 --end 2025-01-12

# 算法复验与 Agent 输入
uv run python .\scripts\validate_equipment_algorithm.py
uv run python .\scripts\equipment_cli.py export-agent-inputs
uv run python .\scripts\validate_equipment_agent_inputs.py
```

Agent 输入位于 `agent_inputs/equipment_health`。`index.json` 是用户索引，`users/<user_id>.json` 包含状态、阶段概率、健康指数、趋势、历史和处置建议，不包含三轴原始数组。首次验证可添加 `--limit-files 10`。

### 安全作业

`safety_operations` 集成 YOLO、ByteTrack、安防规则、豆包视频复核、SQLite 最终决策、Agent 发件箱和操作审计。

在 `.env` 配置 `ARK_API_KEY`、`ARK_MODEL_ID` 和 `SAFETY_AGENT_TOKEN` 后显式运行：

```powershell
uv run python -m intelligent_detection_agent.safety_operations.monitor `
  --config .\safety_operations\config.yaml `
  --source 'E:\path\to\video.mp4'
```

检测结束后，程序最多对同一源视频执行一次豆包复核。确认问题进入 `alert_records` 并通知平台；证据不足进入人工复核列表；排除事件仅保留审计记录。平台未运行时，可在启动后自动补收，或显式执行：

```powershell
uv run python -m intelligent_detection_agent.safety_operations.notifier `
  --config .\safety_operations\config.yaml
```

安全作业页面可以查看异常截图、复核视频、YOLO 规则结果、豆包视觉复核和人员 PPE 结论。确认、处理和关闭操作追加到 `event_handling_actions`；事件关闭后操作人和备注不可编辑。

### 智能问答

侧栏“智能问答”使用单个 DeepAgent，支持：

- 查询用气、智能计量、智能设备和安全作业数据。
- 解析今天、昨天、上周等相对日期。
- 生成包含 ECharts 图表、表格和 SQL 来源的报告。
- 下载包含图表的 HTML 报告，下载不触发额外人工确认。
- 创建计量、设备或安防来源工单，写入前必须人工确认。
- 信息不足时暂停并等待用户补充。

数据库查询使用 `sqlglot` AST、表白名单、只读连接、行数限制、大小限制和超时保护。Agent 没有 Shell 或 Python 执行工具；FilesystemBackend 只读挂载包内的 `conversation_agent/skills`。安防事件在对话中只能查询，确认、处理和关闭必须在安全作业页面完成。

每个用户可以浏览、继续和删除自己的历史对话。消息、报告、SQL、工单和 checkpoint 存储在 `database/user_data.db`，不同用户通过内部线程编号隔离。回答使用 POST + SSE 流式输出；页面展示可折叠 Todo，但不展示工具流水和模型隐藏推理。

报告使用安全的 GFM Markdown 渲染。数据与报告面板默认隐藏，新报告生成后自动展开；每条回答可以独立打开对应报告、SQL 或工单。

### Agent 自动解读

系统设置中的“Agent 自动解读”每次打开页面都默认为关闭，不持久化：

- 关闭时，不自动请求 `/agent/inspect`。
- 手动“生成智能检查结果”仍会调用 LLM。
- 开启后，企业、日期或模块变化时对当前详情自动调用一次。

计量和设备解读使用两个独立的 LangGraph `StateGraph`。后端按企业和日期读取可信诊断结果，再根据风险走不同的固定分支；LLM只负责原因排序、支持/反向证据分析、证据缺口和核查建议，不执行代码，也不能修改算法数值与状态。通过结构校验的报告按“诊断内容 + 模型 + 工作流版本 + 现场补充信息”生成指纹，复用结果保存在 `database/gas_ai_results.duckdb` 的 `inspection.workflow_report` 表中。

## LangSmith

LangSmith 默认关闭。需要追踪时在 `.env` 设置：

```dotenv
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=your_key
LANGSMITH_PROJECT=intelligent-detection-agent
```

可选配置 `LANGSMITH_ENDPOINT`、`LANGCHAIN_HIDE_INPUTS` 和 `LANGCHAIN_HIDE_OUTPUTS`。状态接口只返回 `tracing_enabled`，不会向前端暴露 Key 或完整 Trace。

## 主要 API

### 认证与对话

- `GET /auth/me`
- `POST /auth/register`
- `POST /auth/login`
- `POST /auth/logout`
- `GET /chat/status`
- `GET /chat/threads`
- `GET /chat/threads/{thread_id}`
- `DELETE /chat/threads/{thread_id}`
- `POST /chat/turns`
- `POST /chat/resume`
- `POST /chat/turns/stream`
- `POST /chat/resume/stream`

### 业务接口

- `POST /metering/diagnose`
- `POST /metering/diagnose/batch`
- `GET /metering/results/{user_id}/{diagnosis_date}`
- `GET /metering/work-orders`
- `GET /users/{user_id}/dates`
- `GET /security/events`
- `GET /security/events/{event_id}`
- `POST /security/events/{event_id}/actions`

完整接口以 `http://127.0.0.1:8000/docs` 为准。

## 测试与构建检查

```powershell
# Python
uv run pytest -q

# 前端
npm --prefix frontend run typecheck
npm --prefix frontend test -- --run
npm --prefix frontend run build

# Playwright 需要先由用户显式启动开发服务
npm --prefix frontend run test:e2e
```

## RAG 技术文档检索

RAG 已作为 `search_technical_documents` 工具接入多轮问答，也可以通过命令行独立验证。数据源默认读取：

```text
dataset/doc/用气体超声流量计测量天然气流量/ingest/records.json
```

运行前由用户自行配置以下环境变量：

- `DASHSCOPE_API_KEY`、`DASHSCOPE_WORKSPACE_ID`：百炼 Embedding 与 Rerank。
- 可选 `RAG_QDRANT_URL`、`RAG_COLLECTION_NAME`。

启动本地 Qdrant，并将 203 条记录写入专用 Collection：

```powershell
.\scripts\start_qdrant.ps1
uv run python -m intelligent_detection_agent.rag.cli index --recreate
```

执行单条查询或两条固定示例：

```powershell
uv run python -m intelligent_detection_agent.rag.cli query "超声流量计单向测量应如何安装？"
uv run python -m intelligent_detection_agent.rag.cli demo
```

运行 30 条人工标注检索评测，报告默认写入
`output/rag_eval_results.json`：

```powershell
uv run --env-file .env python -m intelligent_detection_agent.rag.evaluate
```

评测同时统计 RRF 混合召回与 qwen3-rerank 的 Recall@K、Hit@K、
MRR@20、nDCG@K，并保留逐题 Top5 结果，便于定位漏召回。

Agent 按技术文档检索 Skill 将当前问题和必要多轮上下文补全为可独立理解的问题。RAG 管线直接使用该问题执行“1024维向量相似度 + 中文 BM25 → RRF 融合 → qwen3-rerank”，不再额外调用 LLM 改写。Payload 中图片使用相对于文档目录的路径，例如 `images/fig_003/fig_003.png`，不保存机器绝对路径或图片二进制。

## 综合测评

综合测评包含 30 条 RAG 检索用例和 30 条 Agent 端到端用例，其中新增的 10 条 RAG 同时运行两层评测。默认每条执行一次：

```powershell
# 只检查数据库、Qdrant、模型配置和参考查询，不调用模型
uv run --env-file .env python -m intelligent_detection_agent.evaluation --preflight

# 完整运行；也可使用 --suite rag、--suite agent、--case 或 --repeat
uv run --env-file .env python -m intelligent_detection_agent.evaluation --suite all

# 调试确定性规则时可临时关闭独立 LLM 评审
uv run --env-file .env python -m intelligent_detection_agent.evaluation --suite agent --no-judge
```

每次报告写入 `output/evaluation/<run_id>/`，包含检索结果、Agent 逐题 JSONL、汇总 JSON 和 Markdown 报告。被测指标包括任务成功率、LLM 调用轮数、工具次数与准确率、参数准确率、端到端延时、首字延时和 token；评审模型的调用和 token 单独记录。

### 蒸馏筛选与 SFT 导出

评测硬规则只禁止用例明确不允许的工具行为（例如未请求创建工单时调用创建工具）；`allowed_tools` 是 Precision 的预期工具集合，额外查询不自动判任务失败。范围外请求仍禁止业务和 RAG 调用。必要工具、只读 SQL 和目标事实表检查继续保留。

日期接受中文、斜线及 ISO 写法；小数量值采用用例配置的数值容差，支持千位分隔。SQL 不要求固定日期字面量、函数或过滤写法，等价查询、分步查询及返回后筛选由 Judge 根据任务和参考事实核对；关闭 Judge 时不包含这部分语义验证。普通评测中的相关 SQL 用例也会调用 Judge。

轨迹合并仅容忍同一 assistant 发起的完整并行工具返回块内部重排：调用 ID 集合及每个 ID 对应的内容必须完全一致，导出保留首次实际观察到的返回顺序。其他历史改写、缺失、重复或跨步骤移动仍不导出。失败会在 `distillation.trace_error` 中记录首个差异的模型调用序号、消息序号、变化字段、角色及长度，不保存原始消息。旧记录需要重新执行才能验证这项兼容处理；旧评分不能替代新标准的 Judge 评审。

Agent 评测支持 `--concurrency`（默认 1，正整数）。所有“问题 × 重复次数”进入同一个队列，最多同时运行指定数量的完整用例；同一用例内部多轮问答和审批恢复仍按顺序执行。

```powershell
# 30个Agent用例各跑5次，共150个任务，最多30个用例同时执行
uv run --env-file .env python -m intelligent_detection_agent.evaluation `
  --suite agent --repeat 5 --concurrency 30 --distill --distill-threshold 85

# 单个只读问题跑5次，实际并发数为5
uv run --env-file .env python -m intelligent_detection_agent.evaluation `
  --suite agent --case query_01_data_coverage --repeat 5 --concurrency 30 --distill
```

并发上限包括 Agent、Judge 和结果保存，不是严格的 HTTP 请求数上限；工具内部仍可能并行请求。该参数不改变独立 RAG 检索基准的执行方式。普通用例异常记录失败后继续；文件写入等全局错误停止整批并等待活动运行清理。结果 JSONL 和 SFT 按完成顺序写入，使用用例 ID、重复次数、线程 ID 和 SFT 行号对应；Markdown 按用例 ID 和重复次数排序。汇总保存配置并发数、实际 worker 数及整批耗时。模型服务限流可能影响实际吞吐，失败用例不会自动整条重跑。

开启蒸馏后，每个通过硬规则且可完整表示的 Agent 用例使用一次联合 Judge 评审答案和执行过程；入选结果直接写入 `output/evaluation/<run_id>/sft.jsonl`，不另存候选轨迹归档。

```powershell
uv run --env-file .env python -m intelligent_detection_agent.evaluation `
  --suite agent --repeat 3 --distill `
  --distill-threshold 85 --distill-weights 0.4,0.4,0.2

# 只评分，不生成 SFT 文件
uv run --env-file .env python -m intelligent_detection_agent.evaluation --suite agent --distill --no-export-sft

# 使用已有分项结果重新计算，不调用 Agent/Judge，也不修改已导出的 SFT 文件
uv run --env-file .env python -m intelligent_detection_agent.evaluation --regrade output/evaluation/<run_id> --distill --distill-threshold 90
```

- 权重依次为答案、过程、效率，必须非负且总和为 1；阈值范围为 0～100，等于阈值也入选。`--distill` 不能与 `--no-judge` 或纯 `--suite rag` 同用。
- 答案分为 Judge 的 1～5 分乘以 20；过程分为逐次参数正确性 × 60% + 步骤依赖 × 25% + 错误恢复 × 15%。过程子分使用 0/25/50/75/100 五档。
- 效率从 100 分开始，每次超预算调用扣 10 分，每次 Judge 确认的无效重复扣 15 分，同一调用只取较高扣分，最低 0 分。HITL 重放不重复计数，合理重试/必要刷新不算无效重复，但仍计入调用预算。
- 硬规则、答案评审、轨迹完整性和未恢复错误检查是准入条件，不能由高总分抵消。工具 Precision/Recall、Token 与耗时继续报告，不加入加权总分。
- 评审输入超过 120,000 字符时不截断，记录未评审并不入选；评审失败不会停止后续用例。预期的审批/澄清中断不要求最终正文或尚未执行工具的结果。

每个 SFT 行只包含 `messages` 和 `tools`：使用 `system/user/assistant/tool` 角色、函数工具 Schema、对象形式的 `tool_calls[].function.arguments` 和配对的 `tool_call_id`。每行是一整段用例轨迹，保留实际系统提示和工具结果，不包含 `reasoning_content`、推理标签、Judge 评分或标准答案。训练时需使用支持工具调用的学生模型聊天模板，并只对 assistant 消息计算损失；本项目不修改训练代码。

第一版支持文本轨迹。上下文被压缩/改写、工具定义变化、多模态内容或消息断链不能忠实合并时，记录原因并不导出。入选文件逐条刷新后再删除临时 checkpoint；`agent_cases.jsonl` 保存分项分数、模型、评分版本、筛选配置和 SFT 行号，报告分别统计任务成功、评分入选及实际导出。

**不保存候选轨迹意味着：降低阈值只能重新判定已有分数，不能补导出过去未保存的样本；这些用例需要重新执行。** 旧评测缺少过程评分时不会补默认分。正式训练应与评测用例隔离；同一用例的重复轨迹不要跨训练集和测试集分配。

## 当前数据限制

- 历史数据只有 19 天，正常用气预测属于短期稳健基线，不是长期季节性预测。
- 检定记录缺少明确管路号，目前按用户最新一块可关联表具建立误差曲线。
- SCADA 没有出口压力，依赖出口压力的压损诊断无法执行。
- 补气量是算法估算值，必须经过现场核查和企业计量规则确认后才能用于结算。
- 振动退化数据包含合成样本，不能当作现场真实寿命记录。

## 其他资料

- 完整 DuckDB 查询示例：`docs/query_examples.sql`
- SCADA 未识别文件：`reports/failed_scada_files.csv`
