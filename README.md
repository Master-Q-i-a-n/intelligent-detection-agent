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

复制 `.env.example` 为 `.env`，按需填写：

- `DEEPSEEK_API_KEY`、`LLM_BASE_URL`、`LLM_MODEL`：通用大模型配置。
- `CHAT_LLM_*`：智能问答专用模型；未填写时沿用通用配置。
- `ARK_API_KEY`、`ARK_MODEL_ID`：安全作业视频复核。
- `SAFETY_AGENT_TOKEN`：安全作业内部通知鉴权令牌。
- `AUTH_COOKIE_SECURE`：本地 HTTP 使用 `false`，生产 HTTPS 使用 `true`。

### 开发模式

一个终端同时启动 Vite 和 FastAPI：

```powershell
.\start_api.ps1
```

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

`smart_metering.py` 的主要流程：

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

`smart_equipment.py` 使用三轴振动执行多任务健康诊断：

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
uv run python -m safety_operations.monitor `
  --config .\safety_operations\config.yaml `
  --source 'E:\path\to\video.mp4'
```

检测结束后，程序最多对同一源视频执行一次豆包复核。确认问题进入 `alert_records` 并通知平台；证据不足进入人工复核列表；排除事件仅保留审计记录。平台未运行时，可在启动后自动补收，或显式执行：

```powershell
uv run python -m safety_operations.notifier `
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

数据库查询使用 `sqlglot` AST、表白名单、只读连接、行数限制、大小限制和超时保护。Agent 没有 Shell 或 Python 执行工具；FilesystemBackend 只读挂载 `conversation_agent/skills`。安防事件在对话中只能查询，确认、处理和关闭必须在安全作业页面完成。

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
uv run python -m rag.cli index --recreate
```

执行单条查询或两条固定示例：

```powershell
uv run python -m rag.cli query "超声流量计单向测量应如何安装？"
uv run python -m rag.cli demo
```

运行 20 条人工标注检索评测，报告默认写入
`output/rag_eval_results.json`：

```powershell
uv run --env-file .env python -m rag.evaluate
```

评测同时统计 RRF 混合召回与 qwen3-rerank 的 Recall@K、Hit@K、
MRR@20、nDCG@K，并保留逐题 Top5 结果，便于定位漏召回。

Agent 按技术文档检索 Skill 将当前问题和必要多轮上下文补全为可独立理解的问题。RAG 管线直接使用该问题执行“1024维向量相似度 + 中文 BM25 → RRF 融合 → qwen3-rerank”，不再额外调用 LLM 改写。Payload 中图片使用相对于文档目录的路径，例如 `images/fig_003/fig_003.png`，不保存机器绝对路径或图片二进制。

## 当前数据限制

- 历史数据只有 19 天，正常用气预测属于短期稳健基线，不是长期季节性预测。
- 检定记录缺少明确管路号，目前按用户最新一块可关联表具建立误差曲线。
- SCADA 没有出口压力，依赖出口压力的压损诊断无法执行。
- 补气量是算法估算值，必须经过现场核查和企业计量规则确认后才能用于结算。
- 振动退化数据包含合成样本，不能当作现场真实寿命记录。

## 其他资料

- 完整 DuckDB 查询示例：`docs/query_examples.sql`
- SCADA 未识别文件：`reports/failed_scada_files.csv`
