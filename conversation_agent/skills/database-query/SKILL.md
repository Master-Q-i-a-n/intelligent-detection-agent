---
name: database-query
description: 查询燃气用气、智能计量、智能设备和安全作业数据时使用，包含数据库选择、字段、关联关系和指标口径。
---

# 数据库查询

## 必须遵守

1. 相对日期先调用 `get_current_time`。
2. 查询只能调用三个只读 SQL 工具，不得请求文件、Python、Shell 或数据库写入。
3. SQL 必须是单条 `SELECT` 或 `WITH ... SELECT`；DuckDB 表名必须包含 schema。
4. 字段不确定时调用 `describe_data_source`，不要猜字段。
5. 先阅读与任务对应的 reference，读取时传 `limit=1000`。
6. 报告和回答必须区分实测值、算法推导值、估算值与 LLM 解释。
7. DuckDB 禁止使用 `(a, b, ...) IN (SELECT a, b, ...)` 多列元组子查询；每组最新记录必须使用聚合结果 `JOIN`，参考 `references/sql-examples.md`。
8. 查询工具返回 `SQL_QUERY_ERROR` 时，只在 `retry_allowed=true` 时重写整条 SQL 并重试一次，不得原样重试；若再次失败则停止查询并如实说明。

## 数据源选择

- 原始用气、用户档案、表具、检定、维修、振动原始与日级数据：`query_business_data`
  - 参考 `references/gas-input-database.md`
- 智能计量诊断、异常区间、智能设备健康结果、工单：`query_diagnosis_data`
  - 参考 `references/diagnosis-results-database.md`
- 安防事件、LLM 视频复核、通知和处置审计：`query_security_data`
  - 参考 `references/security-database.md`

## 业务口径

- 日用气量、完整度和有效数据：先读 `references/metric-definitions.md`。
- 跨表查询：先读 `references/relationships-and-granularity.md`。
- 常见查询写法：参考 `references/sql-examples.md`，只能借鉴，必须根据用户条件调整。

## 无数据处理

- 自然日期没有数据时明确说明，不得自动替换成最大数据日期。
- 可额外查询 `MIN(data_date)`、`MAX(data_date)` 告知覆盖范围。
- 零行不是信息不足；只有阈值、企业或时间范围不明确且会改变结果时才调用 `ask_user`。
