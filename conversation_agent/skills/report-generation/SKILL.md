---
name: report-generation
description: 用户要求生成用气、计量、设备、安防或综合报告时使用，提供报告计划、模板、图表选择和数据溯源要求。
---

# 报告生成

1. 明确报告对象、时间范围和用途；缺少关键条件时调用 `ask_user`。
2. 读取 `references/report-planning.md`，选择 `templates/` 中最接近的模板。
3. 先读取 database-query Skill 并执行必要 SQL。
4. 所有事实和图表必须引用真实 `query_id`，不得在报告参数中手填或推测数据点。
5. 图表规则见 `references/chart-selection.md`，字段结构见 `references/chart-schema.md`。
6. 调用 `build_report_artifact` 生成前端报告产物；不得写文件。
7. 最终回答说明报告已生成、包含的数据范围和重要数据边界。

报告必须区分：实测采集值、数据库聚合值、算法预测/估算值、LLM 解释。
