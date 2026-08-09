# 图表字段结构

`build_report_artifact.charts` 中每项：

```json
{
  "id": "daily_usage",
  "type": "line",
  "title": "近7日标况用气量",
  "unit": "m³",
  "source_query_id": "qry_xxx",
  "x_field": "data_date",
  "y_fields": ["volume_m3"],
  "series_names": ["标况用气量"]
}
```

支持类型：`line`、`bar`、`pie`、`scatter`。工具会校验查询编号及字段是否真实存在。
