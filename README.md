# 燃气智能检测 Agent：输入数据库

本目录只建立算法所需的原始输入数据，不包含异常诊断、风险分级、补气量、工单等输出结果。

## 数据库结构

一个物理数据库 `database/gas_ai_input.duckdb`，包含三个逻辑库：

1. `asset`：用户与表具档案
2. `inspection`：流量计检定、检定点和维修记录
3. `telemetry`：SCADA时序观测和文件导入日志
4. `equipment`：振动传感器与企业流量计映射
5. `vibration`：三轴振动窗口、五阶段健康标签和退化轨迹

高容量SCADA数据以日期分区的Parquet文件保存在 `dataset/telemetry`，DuckDB中的
`telemetry.scada_observation` 视图提供统一SQL查询入口。

三轴振动数据以日期分区的Parquet文件保存在 `dataset/vibration`，DuckDB中的
`vibration.acceleration_window` 和 `vibration.daily_health` 视图分别提供原始窗口与
Agent友好的每日健康状态查询。该数据由工程师三分类样本派生，`is_synthetic=true`，
不可当作现场真实寿命退化记录。

## 构建命令

仅在需要从原始文件重建数据库时设置数据源目录：

```powershell
$env:GAS_SOURCE_ROOT='D:\path\to\gas-source-data'
$env:GAS_VIBRATION_SOURCE_ROOT='D:\path\to\vibration-source-data'
uv run python .\build_database.py --mode master
uv run python .\build_database.py --mode scada --overwrite
uv run python .\build_vibration_database.py --overwrite
uv run python .\validate_vibration_database.py
```

## 智能设备振动数据

- 企业数：715户；
- 时间范围：2024-12-25至2025-01-12，共19天；
- 三轴窗口：13,585条，每户每天1条；
- 每条窗口：后盖X/Y/Z三轴，每轴500点，采样频率1 kHz；
- 阶段标签：H0健康稳定、H1轻微衰减、H2中度衰减、H3重度衰减、H4故障异常；
- 轨迹类型：健康稳定、缓慢衰减、持续亚健康、加速衰减、突发故障、维修恢复。

## 智能设备健康诊断算法

`smart_equipment.py` 实现三轴振动多任务诊断：

1. 按80/120/160工况执行训练集统计归一化；
2. 通过3/5/9/17四尺度数学形态学开闭残差提取冲击与包络特征；
3. 使用尺度注意力和三轴门控自适应融合不同尺度、不同方向的振动响应；
4. 使用主分类头、CORAL有序辅助头和健康指数回归头联合建模H0-H4阶段顺序；
5. 使用分类—健康指数一致性损失约束阶段概率与0-100健康指数方向一致；
6. 基于连续日期健康指数斜率、最大单日下降及突变前斜率识别稳定、慢衰减、
   快衰减、突发故障和维修恢复趋势；
7. 对连续阶段执行保留突变的稳健平滑和单调退化/恢复约束；
8. 诊断结果写入结果库的 `equipment.health_diagnosis` 与
   `equipment.health_trend`。

训练命令：

```powershell
uv run python .\equipment_cli.py train --epochs 20 --batch-size 256
```

单日诊断和趋势诊断：

```powershell
uv run python .\equipment_cli.py diagnose `
  --user-id 1071586391 --date 2025-01-12

uv run python .\equipment_cli.py trend `
  --user-id 1071586391 --start 2024-12-25 --end 2025-01-12
```

算法复验：

```powershell
uv run python .\validate_equipment_algorithm.py
```

生成全部715家企业的Agent输入JSON：

```powershell
uv run python .\equipment_cli.py export-agent-inputs
uv run python .\validate_equipment_agent_inputs.py
```

Agent输入位于 `agent_inputs/equipment_health`：`index.json` 是用户索引，
`users/<user_id>.json` 包含当前状态、五阶段概率、健康指数、趋势、19天历史、
模型注意力解释和处置建议，不包含三轴原始数组。

首次验证可添加 `--limit-files 10`。完整查询示例见 `query_examples.sql`。

## 当前构建结果

- 用户与表具档案：535户
- 检定记录：3,044条
- 检定流量/误差点：9,570条
- 维修记录：621条
- SCADA时间范围：2024-12-25至2025-01-12，共19天
- SCADA标准化分管路记录：20,794,808条
- 成功读取源文件：14,554个
- 未识别管路字段的源文件：95个（详见 `reports/failed_scada_files.csv`）
- 三类输入均可关联的用户：393户

## 智能计量模块

`smart_metering.py` 已实现以下流程：

1. 从DuckDB读取用户、检定和SCADA数据；
2. 五分钟重采样、异常值处理、缺失填充和完整度评价；
3. 加载既有CWT-Inception-SimAM模型识别用气状态；
4. 交叉验证模型状态与远传计量状态，识别“走气未走字”；
5. 执行压力、温度、双管流量不平衡和传感器异常规则；
6. 使用历史同时间点中位数和MAD建立正常用气基线；
7. 定位实际流量显著低于基线的连续异常区间；
8. 基于检定流量—示值误差曲线和基线缺口估算补气量；
9. 分析表具小流、正常和超量程运行占比；
10. 完成风险评分并自动生成核查工单。

输入数据库保持只读，诊断结果单独写入：

`database/gas_ai_results.duckdb`

结果库包含：

- `metering.diagnosis_run`：每日综合诊断；
- `metering.anomaly_interval`：异常起止时间及缺失量；
- `metering.work_order`：核查工单。

### 命令行诊断

```powershell
uv run python .\metering_cli.py `
  --user-id 2267475 `
  --date 2025-01-12 `
  --output '.\reports\diagnosis_2267475_2025-01-12.json'
```

添加 `--no-model` 可在不加载深度模型时只执行规则、基线和量程分析。

### 启动API

```powershell
uv sync
.\start_api.ps1
```

启动后访问 `http://127.0.0.1:8000/docs`，主要接口包括：

- `POST /metering/diagnose`
- `POST /metering/diagnose/batch`
- `GET /metering/results/{user_id}/{diagnosis_date}`
- `GET /metering/work-orders`
- `GET /users/{user_id}/dates`

### 当前数据限制

- 只有19天历史数据，正常用气预测属于短期稳健基线，尚不是长期季节性预测模型；
- 检定记录缺少明确的管路号，当前按用户最新一块可关联表具建立误差曲线；
- SCADA数据没有出口压力，依赖出口压力的压损诊断无法执行；
- 补气量为算法估算值，必须经过现场核查和企业计量规则确认后才能用于结算。
# 工业AI双模块可视化与检查 Agent

启动 API 后访问 `http://127.0.0.1:8000/`，即可进入“曜衡智控”工业风智能检测平台。页面包含：

- 智能计量：用气量与正常基线、补气量构成、综合风险、表具量程适配、管路数据完整度、异常区间及工单证据。
- 智能设备：19 日健康指数、H0-H4 五阶段概率、三轴振动波形、轴向/形态学尺度权重、状态演化及每日诊断明细。
- 智能检查 Agent：将企业、日期对应的算法结构化结果与现场文字联合生成检查结论、证据链、现场清单和工单建议；关键数值严格沿用算法结果。

Windows 启动方式：

```powershell
cd E:\MyWork\Agent\intelligent-detection-agent
uv sync
.\start_api.ps1
```

若当前项目根目录的 `.env` 中大模型配置可用，Agent 会在本地规则报告上进行受约束的语言增强；接口不可用时自动回退至本地可审计规则，不影响页面检测功能。可复制 `.env.example` 后填写真实密钥。
