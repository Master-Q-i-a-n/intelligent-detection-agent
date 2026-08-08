from __future__ import annotations

from typing import Any

import pandas as pd


class DataQualityAnalyzer:
    """项目内置的 SCADA 数据质量处理器，用于替代已丢失的旧版外部模块。"""

    def process_user_data(self, data: pd.DataFrame, user_name: str) -> dict[str, Any] | None:
        if data is None or data.empty:
            return None

        frame = data.copy()
        frame.index = pd.DatetimeIndex(pd.to_datetime(frame.index, errors="coerce"))
        frame = frame[~frame.index.isna()].sort_index()
        frame = frame[~frame.index.duplicated(keep="last")]
        if frame.empty:
            return None

        for column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

        # 统一到当天 288 个五分钟时刻，短缺口插值，长缺口保持缺失以便计算完整度。
        day_start = frame.index.min().normalize()
        full_index = pd.date_range(day_start, periods=288, freq="5min")
        processed = frame.resample("5min").mean().reindex(full_index)
        result: dict[str, Any] = {"用户": str(user_name), "处理后数据": processed}
        usable_pipelines = 0

        for pipeline_no in range(1, 5):
            flow_column = f"{pipeline_no}号标况瞬时"
            pipeline_columns = [column for column in processed.columns if column.startswith(f"{pipeline_no}号")]
            if not pipeline_columns:
                result[f"管道{pipeline_no}完整度"] = 0.0
                result[f"管道{pipeline_no}用气量"] = 0.0
                continue

            reference_column = flow_column if flow_column in processed.columns else pipeline_columns[0]
            completeness = float(processed[reference_column].notna().mean())
            result[f"管道{pipeline_no}完整度"] = completeness
            if completeness >= 0.5:
                usable_pipelines += 1

            for column in pipeline_columns:
                series = processed[column]
                if column.endswith("瞬时"):
                    processed[column] = series.interpolate(limit=2).clip(lower=0)
                else:
                    processed[column] = series.interpolate(limit=3, limit_direction="both")

            flow = processed[flow_column] if flow_column in processed.columns else pd.Series(0.0, index=full_index)
            result[f"管道{pipeline_no}用气量"] = float(flow.fillna(0).sum() * 5.0 / 60.0)

        # 旧接口以 0 表示数据有效；至少一条管路达到 50% 完整度才进入规则诊断。
        result["是否有效"] = 0 if usable_pipelines else 2
        result["处理后数据"] = processed
        return result
