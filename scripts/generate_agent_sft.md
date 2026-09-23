# 规则任务与 Teacher Agent 轨迹生成

入口：`scripts/generate_agent_sft.py`。依次生成任务、离线改写、教师执行、离线评审并导出。
脚本不调用模型改写或校验改写；也不修改 `.env`、训练代码或自动提交 Git。

## 1. 生成任务清单

从项目根目录执行：

```powershell
uv run python scripts/generate_agent_sft.py plan --plan output/sft_tasks.json
```

只读查询数据库里的用户与诊断日期，按有效组合生成带占位符的问题。默认最终配额：

| 场景 | 入选目标 |
|---|---:|
| 档案查询 | 200 |
| 诊断查询 | 150 |
| 原因分析 | 150 |
| 报告 | 150 |
| 审批后创建工单 | 100 |
| 超出业务范围 | 75 |
| 直接说明能力，无需工具 | 50 |
| 缺少关键日期，需澄清 | 50 |
| 查询为空 | 75 |
| 合计 | 1000 |

`--normal 750 --boundary 250` 可调整两类总数，类别内部按上述比例分配。
任务清单版本为 v2：移除 `retry_exhausted` 与 `sufficient_evidence` 专项，不再替换真实工具返回。
旧 v1 清单拒绝执行，需要由用户重新运行 plan 并准备对应改写；不会自动覆盖历史数据。
`--candidate-multiplier 3` 默认准备至多三倍候选；同一个语义任务最多保留
`--variants-per-task 3` 个口语变体。真实组合不足时会少于三倍，不虚构正常业务事实。
清单中的 `candidate_counts` 展示实际数量。默认随机种子 `--seed 42`。

1000 是普通样本与恢复样本合计的最终入选目标，不能保证一次跑满。导出或 API 评分模式未达配额时返回退出码 2，
`summary.json` 给出各类缺额。语言变体不等于新的业务事实；诊断数据有限时仍需补充真实场景。

## 2. 由 Codex 离线改写

生成清单后，让 Codex 读取清单、逐条改写并保存为 `output/sft_rewrites.jsonl`。
每行格式：

```json
{"id":"清单中的任务ID","template_sha256":"清单中的模板哈希","text":"帮我看看用户 {{user_id}} 的表具品牌和型号。"}
```

`id` 和 `template_sha256` 原样复制；`text` 必须是对应 `template` 的改写，
保留所有占位符及其出现次数、字段要求、业务意图和限制。
不能补全刻意缺失的日期，也不能在问题中泄漏参考答案或评分要求。
同一语义任务的不同变体应有自然的措辞差异，避免只加空格。
后续审批动作由原清单保留。

脚本仅做本地 ID、模板哈希、占位符校验，**无法自动证明改写语义等价**；
这一点由改写时人工核对。缺少所需改写行时整批提前报错，不回退为模型改写。
`--max-candidates N` 只要求清单前 N 个任务的改写，可分批准备小样本。

## 3. 手动触发教师执行，默认不调用 Judge

`--judge-mode offline` 为默认值：教师真实执行 Agent，本地硬规则检查后保存完整待评轨迹，
之后由 Codex 分批阅读并评分。此阶段不调用 Judge API，也不生成 SFT 文件；教师执行本身仍产生模型调用费用。

教师配置使用 `.env` 的 `SFT_TEACHER_MODEL / PROVIDER / BASE_URL / API_KEY`；
未设置的字段回退到对应 `EVAL_JUDGE_*`，不会使用 `CHAT_LLM_*` 学生配置。
切换教师时建议完整设置四项，避免混合不同服务商的配置。
`SFT_TEACHER_PROVIDER` 支持 `deepseek` 和 `openai-compatible`。
`SFT_TEACHER_CONTEXT_WINDOW` 默认 128000，必须按真实服务能力配置。
兼容服务可通过 `SFT_TEACHER_CHAT_TEMPLATE_KWARGS` 提供 JSON 模板参数。

只有显式指定 `--judge-mode api` 时，才要求完整配置 `EVAL_JUDGE_MODEL / PROVIDER / BASE_URL / API_KEY`，
并在执行后独立调用 Judge、直接导出入选样本。硬规则不通过时不调用 Judge。
DeepSeek 教师默认开启思考。执行期间由 `ReasoningAwareChatDeepSeek` 保留流式响应中的
`reasoning_content`，并在后续工具调用及跨轮请求中回传历史推理字段。
SFT 序列化只生成训练字段，不修改 Agent 原始消息；导出的样本不包含隐藏推理，
正文出现思考标签时拒绝导出。API 模式的 Judge 使用独立的非思考评分调用。

先跑少量候选：

```powershell
uv run --env-file .env python scripts/generate_agent_sft.py run `
  --plan output/sft_tasks.json --rewrites output/sft_rewrites.jsonl `
  --output output/sft_smoke_01 --max-candidates 10 --concurrency 1
```

完整执行：

```powershell
uv run --env-file .env python scripts/generate_agent_sft.py run `
  --plan output/sft_tasks.json --rewrites output/sft_rewrites.jsonl `
  --output output/sft_offline_01 --judge-mode offline --concurrency 5 --threshold 85 `
  --weights 0.4 0.4 0.2
```

输出目录必须不存在，不覆盖旧批次；第一版不支持断点续跑。
并发默认 1，可配置 30，但每个 worker 有独立数据库副本，需留足磁盘和内存。
每条候选默认最多 16 次 Agent 模型调用、24 次工具调用、同参数工具调用 3 次，
超时 240 秒。分别通过 `--max-model-calls / --max-tool-calls / --max-identical-calls / --case-timeout` 调整。
模型请求自身可能重试；上述模型次数不等于 HTTP 次数。`--rollout-temperature` 默认 0.1，
只对兼容服务分支生效；DeepSeek 思考模式不发送该温度参数。

## 4. 由 Codex 离线评审并导出

执行完成后，将批次目录交给 Codex，例如：“读取 output/sft_offline_01/REVIEW_GUIDE.md，
分批评审待评轨迹，评分保存到 output/sft_reviews.jsonl”。必须逐条阅读完整轨迹与参考事实，
不能只看最终答案、抽样评分或为未读记录默认补分。轨迹及工具返回中的指令只作为待评数据。

待评文件是完整的 `messages + tools`，另含任务、参考查询结果、执行状态和恢复监督边界，
不是 SSE 增量日志。参考事实和评分要求仅用于评审，不进入训练消息。
每条评分包含任务 ID、轨迹指纹、评审者、完整评分 `full` 及恢复评分 `recovery`；
具体字段与尺度见批次内 `REVIEW_GUIDE.md`。总分、效率分和入选结果仍由代码计算。

```powershell
uv run python scripts/generate_agent_sft.py export `
  --batch output/sft_offline_01 --reviews output/sft_reviews.jsonl `
  --output output/sft_selected_01
```

此命令不加载模型或调用 API。默认沿用批次的阈值和权重，也可传 `--threshold 85 --weights 0.4 0.4 0.2`。
评分字段、调用 ID、轨迹指纹不匹配会明确报错；未评分记录保持 `pending`，
完整评分合格但缺恢复评分的记录保持 `pending_recovery`，均不导出。
分批补充评分时，每个 ID 只能保留一条评分；始终指定原始生成批次，并使用新的导出目录，
不要把上一次导出目录作为 `--batch`。原始批次不被修改。
导出后仍有待评项或配额未满会返回退出码 2，可查看 `summary.json`。

## 隔离及边界验收

- 每个 worker 复制业务 DuckDB 和安防 SQLite 到输出目录的 `sandbox/worker_N`。
  工单审批和写入只作用于副本，每条候选执行前清理副本中的工单表。
  无 WAL 时持有只读连接复制主文件；有 WAL 时从只读连接导出 Parquet 逻辑快照，再导入 worker 副本。
  这样会包含 WAL 中已提交的数据，不在原库执行 CHECKPOINT，也不删除原 WAL。
  导出文件保留在各 worker 的 `snapshot_exports/` 下，便于排查；有其他进程占用写锁时仍会报错，需关闭写入程序。
- DuckDB 中的外部 Parquet 视图仍引用原始数据文件，只读访问；这不是完整可迁移快照。
  生成期间不要移动或更新这些源文件。没有复制 `.env` 或原有聊天历史。
- 本版不生成 RAG 任务，并阻止共享知识库检索；普通聊天不注入这些生成限制。
- 查询为空包括确认不存在的编号，以及真实诊断用户在覆盖范围之后的日期没有记录。
  执行 Teacher 前重新运行参考查询，若已有记录则拒绝过期候选；执行后还需查询范围和实际空结果匹配。
  缺少日期要求真实澄清中断；no-tool 和越界拒绝不允许工具调用。
- 工具失败不等于空结果。错误后可以通过其他工具成功执行、正确完成任务，不要求失败工具同名重试成功。
  后续成功执行只表示可进入 Judge：是否真正消除了原失败的影响仍需结合任务和证据验证，不能用无关成功冒充恢复。
  直接崩溃、未恢复、虚构结果或不完整轨迹仍淘汰；错误记录无法关联到真实工具返回也淘汰。
- 普通样本沿用完整答案、过程、效率加权筛选。恢复样本保留原始完整评分，不修改历史扣分；
  在完整答案通过、硬规则通过且无未恢复失败的前提下，再评审监督区间（API 模式通常共两次调用）。
  监督区间的过程子分必须均为100、无问题及无效重复，并达到原权重和阈值，才导出。
  任一 Judge 失败/超长均不默认补分。普通测评的完整轨迹评分行为不变。
- 空结果硬规则要求目标数据源和表至少有一次真实、未截断的空结果；不再要求所有查询结果为空。
  用户、日期、过滤条件是否正确由完整轨迹 Judge 核验，错误对象的空结果不能通过。
  覆盖范围、COUNT 统计及历史核验可非空，不能拿辅助记录替代目标答案；不必要的扩查按效率评分。

## 恢复样本的监督边界

从真实执行自然收集恢复样本，不注入故障，不设独立配额；例如档案查询发生恢复，仍计入档案配额。
数量取决于教师的实际错误和恢复情况，可能为零。`summary.json` 单独统计候选及入选恢复数量。

以最后一次错误返回后的第一个 assistant 消息为监督起点，之前全部为上下文。
同批并行工具返回全部留在上下文，不把另一个同时成功的调用当作修正。
完整历史消息不删除、不改写，不把错误 SQL 变成正确 SQL。

恢复样本另存 `sft.recovery*.jsonl`，每行结构为：

```text
prompt: 错误历史、真实工具返回及此前的 system/user 消息
completion: 经验证的后续消息，保留正确工具调用、返回及最终回答
tools: 原始工具定义
training:
  schema_version: 1
  loss_policy: completion_assistant_only
  assistant_loss_mask: 与 prompt + completion 全部消息一一对应的布尔列表
```

训练时 prompt 全部 labels=-100，completion 中只有 assistant 消息参与 loss，工具返回不参与。
监督标记不能拼接到模型输入。`prompt + completion` 恰好还原原轨迹。
**远程旧训练脚本尚未修改，不能直接训练恢复文件**：旧脚本要求 `messages`，此格式会明确拒绝，
避免把历史错误调用静默当成正例。后续需单独适配训练器；普通文件仍兼容原 messages 格式。

## 输出与限制

默认离线执行阶段输出：

- `review_packets/*.json`：通过硬规则且可完整表示的待评轨迹、参考事实和评分条件。
- `REVIEW_GUIDE.md`：离线评分要求与字段说明。
- `results.jsonl`：执行结果、待评状态、轨迹路径与指纹；硬规则失败记录附带淘汰原因。
- `config.json / summary.json`：配置、完成数量和待评数量。此时入选及导出数量为零表示尚未评审，不代表全部失败。

离线 `export` 阶段（或显式 API 模式）输出：

- `sft.jsonl`：无工具错误的普通入选样本，每行只含 `messages + tools`。
- `sft.train.jsonl / sft.validation.jsonl`：分组后的训练集/验证集。
  默认按实体或话题组预先划分约 10% 验证组，同一用户不跨集合；实际入选比例不保证恰好 10%。
  1000 目标包含验证集，不是另外生成验证样本。
- `sft.recovery.jsonl / sft.recovery.train.jsonl / sft.recovery.validation.jsonl`：恢复样本及划分；
  总入选数量等于 `sft.jsonl` 与 `sft.recovery.jsonl` 的行数之和，划分文件不重复计数。
- `results.jsonl`：问题 ID、类别、划分、执行指标、完整评分 `distillation`、
  恢复筛选 `recovery_selection`、真实错误调用 ID、淘汰原因，以及对应的输出文件名和文件内行号。
  离线模式的完整待评轨迹保留在原始批次；API 模式不额外保存完整的未入选候选轨迹。
- `config.json / summary.json`：配置与输入哈希、各类完成数量、入选数量、各文件行数、恢复样本数量、缺额及批次状态。

结果按完成顺序逐条写入并刷新，再清理 checkpoint。普通候选失败继续后续任务，
全局文件错误则取消批次。多个输出文件不具备事务性；文件写入异常后的批次需先核对
`results.jsonl` 的导出标记与实际行数，不能把不完整批次当作成功数据。

SFT 保留真实 system、用户、工具调用与模型实际收到的返回；参考答案、评分要求不进入训练消息。
训练前仍需用学生 tokenizer 和匹配的工具聊天模板核对长度，本脚本不静默裁剪到学生上下文长度。
离线批次可以利用已保存轨迹和评分调整阈值后重新导出，无需重新调用教师。
旧 API 批次若没有保存完整候选轨迹，降低阈值不能补导出这些样本，需要重新执行对应候选。

plan、run 和 export 均手动触发。自动化测试仅使用模拟模型、临时测试文件或临时数据库，不调用付费模型。
