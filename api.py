from __future__ import annotations

import json
import math
import os
import secrets
import threading
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional


ROOT = Path(__file__).resolve().parent

import duckdb
import numpy as np
from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from inspection_agent import InspectionAgent
from daily_dashboard import DailyDiagnosisDashboard
from smart_metering import SmartMeteringService, long_to_wide
from safety_operations.db import (
    accept_agent_notification,
    accept_local_pending_notifications,
    connect_database as connect_security_database,
    get_security_event,
    get_security_evidence_path,
    handle_security_event,
    list_security_events,
    security_overview,
)


FRONTEND_ROOT = ROOT / "frontend" / "dist"
EQUIPMENT_INDEX = ROOT / "agent_inputs" / "equipment_health" / "index.json"
EQUIPMENT_USERS = ROOT / "agent_inputs" / "equipment_health" / "users"
INPUT_DB = ROOT / "database" / "gas_ai_input.duckdb"
SAFETY_ROOT = ROOT / "safety_operations"
SAFETY_DB = SAFETY_ROOT / "data" / "security.db"
SAFETY_OUTPUT_ROOT = (SAFETY_ROOT / "outputs").resolve()


def _refresh_parquet_views() -> None:
    """启动时将 DuckDB 视图重新绑定到当前项目目录，支持项目整体移动。"""
    telemetry_glob = (ROOT / "dataset" / "telemetry" / "observation_date=*" / "*.parquet").as_posix()
    vibration_glob = (ROOT / "dataset" / "vibration" / "observation_date=*" / "*.parquet").as_posix()
    with duckdb.connect(str(INPUT_DB)) as con:
        con.execute(
            f"""
            CREATE OR REPLACE VIEW telemetry.scada_observation AS
            SELECT *, CAST(observation_date AS DATE) AS data_date
            FROM read_parquet('{telemetry_glob}', hive_partitioning=1)
            """
        )
        con.execute(
            f"""
            CREATE OR REPLACE VIEW vibration.acceleration_window AS
            SELECT *, CAST(observation_date AS DATE) AS data_date
            FROM read_parquet('{vibration_glob}', hive_partitioning=1)
            """
        )
        con.execute(
            """
            CREATE OR REPLACE VIEW vibration.daily_health AS
            SELECT window_id,user_id,company_name,meter_id,sensor_id,data_date,
                   operating_condition,coarse_label,stage_label,stage_name,health_index,
                   trend_label,trajectory_type,trajectory_name,label_source,is_synthetic
            FROM vibration.acceleration_window
            """
        )


_refresh_parquet_views()

app = FastAPI(title="燃气计量与设备健康智能检测平台", version="1.0.0")
service = SmartMeteringService(use_deep_model=True)
fast_service = SmartMeteringService(use_deep_model=False)
inspection_agent = InspectionAgent(ROOT)
daily_dashboard_service = DailyDiagnosisDashboard(ROOT)
diagnosis_lock = threading.RLock()
diagnosis_cache: Dict[str, Dict[str, Any]] = {}
DETAIL_CACHE_ROOT = ROOT / "reports" / "metering_detail_cache_v2"
DETAIL_CACHE_ROOT.mkdir(parents=True, exist_ok=True)


def _detail_cache_path(user_id: str, diagnosis_date: date) -> Path:
    safe_user = str(user_id).replace("/", "").replace("\\", "")
    return DETAIL_CACHE_ROOT / str(diagnosis_date) / f"{safe_user}.json"


def _repair_text(value: Any) -> Any:
    """修复历史导入阶段由 UTF-8/GBK 误解码形成的中文乱码。"""
    if not isinstance(value, str):
        return value
    repaired = value
    for _ in range(2):
        try:
            candidate = repaired.encode("gbk").decode("utf-8")
        except (UnicodeEncodeError, UnicodeDecodeError):
            break
        if candidate == repaired:
            break
        repaired = candidate
    return repaired.rstrip("?")


class DiagnosisRequest(BaseModel):
    user_id: str
    diagnosis_date: date
    save: bool = True
    deep_model: bool = False


class BatchDiagnosisRequest(BaseModel):
    user_ids: List[str]
    diagnosis_date: date
    save: bool = True


class InspectionRequest(BaseModel):
    module: str
    user_id: str
    diagnosis_date: date
    field_text: str = ""
    context: Dict[str, Any]


class SecurityNotification(BaseModel):
    schema_version: int = 1
    alert_id: str
    notification_kind: Literal["CONFIRMED_ALERT", "REVIEW_REQUIRED"]
    event_id: str
    source_system: str = "yolo_track"
    final_decision: Literal["CONFIRMED", "UNCERTAIN"]
    event_type: str
    camera_id: str
    zone_id: Optional[str] = None
    primary_track_id: Optional[int] = None
    activated_video_seconds: Optional[float] = None
    occurred_at: Optional[str] = None
    severity: Optional[str] = None
    final_reason: Optional[str] = None
    recommended_action: Optional[str] = None
    evidence: List[Dict[str, Any]] = Field(default_factory=list)


class SecurityActionRequest(BaseModel):
    action: Literal["ACKNOWLEDGE", "START_PROCESSING", "CLOSE"]
    operator: str
    comment: Optional[str] = None


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"文件不存在：{path.name}")
    return json.loads(path.read_text(encoding="utf-8"))


def _json_safe(value: Any) -> Any:
    """将算法结果中的 NumPy/Pandas 类型递归转换成 FastAPI 可序列化类型。"""
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, int):
        return value
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, np.ndarray)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "to_pydatetime"):
        return value.to_pydatetime().isoformat()
    if hasattr(value, "item"):
        try:
            return _json_safe(value.item())
        except (TypeError, ValueError):
            pass
    return str(value)


def _equipment_user(user_id: str) -> dict[str, Any]:
    safe_id = str(user_id).replace("/", "").replace("\\", "")
    return _load_json(EQUIPMENT_USERS / f"{safe_id}.json")


def _equipment_for_date(payload: dict[str, Any], diagnosis_date: date) -> dict[str, Any]:
    target = str(diagnosis_date)
    history = payload.get("daily_history", [])
    selected = next((item for item in history if item.get("date") == target), None)
    if selected is None:
        raise HTTPException(status_code=404, detail=f"该企业在 {target} 无设备数据")
    stage = selected.get("temporal_stage") or selected.get("stabilized_stage") or selected.get("predicted_stage")
    health_index = selected.get("stabilized_health_index", selected.get("predicted_health_index"))
    stage_names = {"H0": "健康稳定", "H1": "轻度衰减", "H2": "性能衰减", "H3": "严重衰减", "H4": "故障状态"}
    risk_map = {"H0": "低风险", "H1": "较低风险", "H2": "中风险", "H3": "高风险", "H4": "严重风险"}
    trend_names = {
        "stable": "健康稳定", "slow_decay": "缓慢衰减", "fast_decay": "加速衰减",
        "abrupt_fault": "突发故障", "recovery": "维修恢复",
    }
    action_map = {
        "H0": "设备状态稳定，保持常规监测。",
        "H1": "缩短巡检周期，关注健康指数变化。",
        "H2": "提高采集频次，纳入预防性维护清单。",
        "H3": "建议24小时内复测并安排计划检修。",
        "H4": "立即生成高优先级工单，检查表体、轴承、传动及传感器安装状态。",
    }
    current = {
        "date": target,
        "stage": stage,
        "stage_name": stage_names.get(stage, stage),
        "health_index": health_index,
        "confidence": selected.get("confidence"),
        "probabilities": selected.get("probabilities", {}),
        "risk_level": risk_map.get(stage, "待判定"),
        "operating_condition": selected.get("operating_condition"),
    }
    recent = [item for item in history if item.get("date", "") <= target][-7:]
    recent_values = np.asarray([
        float(item.get("stabilized_health_index", item.get("predicted_health_index", 0))) for item in recent
    ], dtype=float)
    recent_slope = float(np.polyfit(np.arange(len(recent_values)), recent_values, 1)[0]) if len(recent_values) > 1 else 0.0
    recent_drops = recent_values[:-1] - recent_values[1:] if len(recent_values) > 1 else np.asarray([0.0])
    recent_max_drop = float(max(0.0, recent_drops.max()))
    recent_trend = "fast_decay" if recent_slope <= -1.0 or recent_max_drop >= 12 else "slow_decay" if recent_slope <= -0.35 else "stable"
    recent_stages = []
    for item in recent:
        item_stage = item.get("temporal_stage") or item.get("stabilized_stage") or item.get("predicted_stage")
        if not recent_stages or recent_stages[-1] != item_stage:
            recent_stages.append(item_stage)
    return {
        "entity": payload.get("entity", {}),
        "current_assessment": current,
        "trend_assessment": {
            "trend_label": recent_trend,
            "trend_name": trend_names.get(recent_trend, "状态变化"),
            "daily_slope": round(recent_slope, 4),
            "maximum_daily_drop": round(recent_max_drop, 4),
            "health_index_start": round(float(recent_values[0]), 2) if len(recent_values) else health_index,
            "health_index_end": round(float(recent_values[-1]), 2) if len(recent_values) else health_index,
            "stage_sequence": recent_stages,
            "window_days": len(recent),
        },
        "model_explanation": {
            "axis_weights": selected.get("axis_weights", payload.get("model_explanation", {}).get("axis_weights", [])),
            "morphological_scale_weights": selected.get("scale_weights", payload.get("model_explanation", {}).get("morphological_scale_weights", [])),
            "scale_kernel_sizes": payload.get("model_explanation", {}).get("scale_kernel_sizes", [3, 5, 9, 17]),
        },
        "recommended_action": action_map.get(stage, "结合现场情况复核。"),
        "daily_history": history,
        "model": payload.get("model", {}),
        "data_notice": payload.get("data_notice", {}),
    }


@app.get("/health")
def health():
    return {"status": "ok", "modules": ["smart_metering", "smart_equipment", "inspection_agent", "safety_operations"]}


@app.get("/api/users")
def users(limit: int = 1000):
    payload = _load_json(EQUIPMENT_INDEX)
    equipment_items = {str(item["user_id"]): item for item in payload.get("users", [])}
    with duckdb.connect(str(INPUT_DB), read_only=True) as con:
        metering_rows = con.execute(
            """
            SELECT s.user_id, ANY_VALUE(u.station_name) AS station_name,
                   MIN(s.data_date) AS start_date, MAX(s.data_date) AS end_date
            FROM telemetry.scada_observation s
            LEFT JOIN asset.user_meter u ON u.user_id=s.user_id
            GROUP BY s.user_id
            ORDER BY s.user_id
            """
        ).fetchall()
    items = []
    for user_id, station_name, start_date, end_date in metering_rows:
        equipment = equipment_items.get(str(user_id))
        if equipment is None:
            continue
        item = dict(equipment)
        item["company_name"] = _repair_text(station_name or equipment.get("company_name") or str(user_id))
        item["date_range"] = [str(start_date), str(end_date)]
        items.append(item)
        if len(items) >= max(1, min(limit, 2000)):
            break
    return {
        "items": items,
        "count": len(items),
        "enterprise_count": len(items),
        "date_range": payload.get("date_range", []),
    }


@app.get("/daily/overview/{diagnosis_date}")
def daily_overview(diagnosis_date: date):
    return daily_dashboard_service.overview(diagnosis_date)


@app.get("/daily/metering-history/{user_id}/{diagnosis_date}")
def daily_metering_history(user_id: str, diagnosis_date: date, days: int = 7):
    return {"user_id": user_id, "items": daily_dashboard_service.metering_history(user_id, diagnosis_date, max(1, min(days, 30)))}


@app.get("/metering/signals/{user_id}/{diagnosis_date}")
def metering_signals(user_id: str, diagnosis_date: date, max_points: int = 1000):
    """返回当天各管路流量、压力、温度曲线，供诊断证据可视化。"""
    day_long = service.repo.get_day_long(user_id, diagnosis_date)
    if day_long.empty:
        raise HTTPException(status_code=404, detail="该企业当天无SCADA曲线数据")
    wide = long_to_wide(day_long)
    user = service.repo.get_user(user_id)
    quality = service.quality_analyzer.process_user_data(wide, user.get("station_name") or str(user_id))
    frame = quality.get("处理后数据") if quality else None
    if frame is None or frame.empty:
        raise HTTPException(status_code=422, detail="当天数据未通过质量处理，无法生成诊断曲线")
    times = list(frame.index)
    # 默认保留当天重采样后的全部时刻；仅在调用方显式限制时才抽稀。
    step = max(1, math.ceil(len(times) / max(24, min(max_points, 2000))))
    selected_times = times[::step]
    selected = frame.loc[selected_times].copy()
    payload: Dict[str, Any] = {
        "user_id": str(user_id),
        "diagnosis_date": str(diagnosis_date),
        "times": [str(item)[11:16] for item in selected_times],
        "resample_frequency": "5min",
        "point_count": len(selected_times),
        "pipelines": {},
    }
    for pipeline_no in range(1, 5):
        flow_col, pressure_col, temperature_col = (
            f"{pipeline_no}号标况瞬时", f"{pipeline_no}号压力", f"{pipeline_no}号温度"
        )
        if not any(column in selected.columns for column in (flow_col, pressure_col, temperature_col)):
            continue
        flow = selected[flow_col] if flow_col in selected.columns else None
        pressure = selected[pressure_col] if pressure_col in selected.columns else None
        temperature = selected[temperature_col] if temperature_col in selected.columns else None
        payload["pipelines"][str(pipeline_no)] = {
            "flow": flow.fillna(0).astype(float).round(4).tolist() if flow is not None else [0.0] * len(selected),
            "pressure": pressure.astype(float).round(4).where(pressure.notna(), None).tolist() if pressure is not None else [None] * len(selected),
            "temperature": temperature.astype(float).round(4).where(temperature.notna(), None).tolist() if temperature is not None else [None] * len(selected),
        }
    return _json_safe(payload)


@app.get("/users/{user_id}/dates")
def available_dates(user_id: str):
    try:
        dates = service.repo.available_dates(user_id)
    except Exception:
        dates = [item["date"] for item in _equipment_user(user_id).get("daily_history", [])]
    return {"user_id": user_id, "dates": [str(item) for item in dates]}


@app.post("/metering/diagnose")
def diagnose(request: DiagnosisRequest):
    cache_key = f"{request.user_id}:{request.diagnosis_date}"
    disk_cache = _detail_cache_path(request.user_id, request.diagnosis_date)
    try:
        # 详情页反复打开时直接复用同一进程内的诊断结果，避免重复加载模型和查询30天数据。
        if not request.save and cache_key in diagnosis_cache:
            return diagnosis_cache[cache_key]
        if not request.save and disk_cache.exists():
            cached = _load_json(disk_cache)
            diagnosis_cache[cache_key] = cached
            return cached
        # DuckDB为单文件数据库，同一日期重复删除/插入必须串行执行。
        with diagnosis_lock:
            if not request.save and cache_key in diagnosis_cache:
                return diagnosis_cache[cache_key]
            diagnosis_service = service if request.deep_model else fast_service
            result = _json_safe(diagnosis_service.diagnose(request.user_id, request.diagnosis_date, request.save))
            diagnosis_cache[cache_key] = result
            disk_cache.parent.mkdir(parents=True, exist_ok=True)
            disk_cache.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
            return result
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/metering/diagnose/batch")
def diagnose_batch(request: BatchDiagnosisRequest):
    outputs = []
    for user_id in request.user_ids:
        try:
            with diagnosis_lock:
                result = _json_safe(service.diagnose(user_id, request.diagnosis_date, request.save))
                diagnosis_cache[f"{user_id}:{request.diagnosis_date}"] = result
                outputs.append(result)
        except Exception as exc:
            outputs.append({"user_id": user_id, "status": "failed", "error": str(exc)})
    return {"diagnosis_date": request.diagnosis_date, "results": outputs}


@app.get("/metering/results/{user_id}/{diagnosis_date}")
def get_result(user_id: str, diagnosis_date: date):
    result = service.get_saved_result(user_id, diagnosis_date)
    if result is None:
        raise HTTPException(status_code=404, detail="未找到诊断结果")
    return _json_safe(result)


@app.get("/metering/work-orders")
def work_orders(status: Optional[str] = None, limit: int = 100):
    return {"items": service.list_work_orders(status, limit)}


@app.get("/equipment/dashboard/{user_id}/{diagnosis_date}")
def equipment_dashboard(user_id: str, diagnosis_date: date):
    return _equipment_for_date(_equipment_user(user_id), diagnosis_date)


@app.get("/equipment/waveform/{user_id}/{diagnosis_date}")
def equipment_waveform(user_id: str, diagnosis_date: date, points: int = 180):
    points = max(50, min(points, 500))
    with duckdb.connect(str(INPUT_DB), read_only=True) as con:
        row = con.execute(
            """
            SELECT accel_x, accel_y, accel_z
            FROM vibration.acceleration_window
            WHERE user_id=? AND data_date=CAST(? AS DATE)
            LIMIT 1
            """,
            [str(user_id), str(diagnosis_date)],
        ).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="未找到三轴振动波形")
    length = len(row[0])
    indices = sorted(set(round(i * (length - 1) / (points - 1)) for i in range(points)))
    return {
        "sampling_rate_hz": 1000,
        "indices": indices,
        "x": [round(float(row[0][i]), 6) for i in indices],
        "y": [round(float(row[1][i]), 6) for i in indices],
        "z": [round(float(row[2][i]), 6) for i in indices],
    }


@app.post("/agent/inspect")
def inspect(request: InspectionRequest):
    try:
        return inspection_agent.generate(
            module=request.module,
            user_id=request.user_id,
            diagnosis_date=str(request.diagnosis_date),
            field_text=request.field_text,
            context=request.context,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


def _security_connection():
    return connect_security_database(SAFETY_DB, 5000)


@app.post("/internal/security/events")
def receive_security_event(
    notification: SecurityNotification,
    authorization: Optional[str] = Header(default=None),
):
    expected = os.getenv("SAFETY_AGENT_TOKEN", "").strip()
    if not expected:
        raise HTTPException(status_code=503, detail="未配置 SAFETY_AGENT_TOKEN")
    supplied = (authorization or "").removeprefix("Bearer ").strip()
    if not supplied or not secrets.compare_digest(supplied, expected):
        raise HTTPException(status_code=401, detail="安防通知鉴权失败")
    connection = _security_connection()
    try:
        with connection:
            return accept_agent_notification(connection, notification.model_dump())
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    finally:
        connection.close()


@app.get("/security/overview")
def get_security_overview():
    connection = _security_connection()
    try:
        with connection:
            accept_local_pending_notifications(connection)
        return security_overview(connection)
    finally:
        connection.close()


@app.get("/security/events")
def get_security_events(
    decision: Optional[str] = None,
    handling_status: Optional[str] = None,
    event_type: Optional[str] = None,
    camera_id: Optional[str] = None,
    after_sequence: Optional[int] = None,
    limit: int = 100,
):
    connection = _security_connection()
    try:
        with connection:
            accept_local_pending_notifications(connection)
        return {"items": list_security_events(
            connection,
            decision=decision,
            handling_status=handling_status,
            event_type=event_type,
            camera_id=camera_id,
            after_sequence=after_sequence,
            limit=limit,
        )}
    finally:
        connection.close()


@app.get("/security/events/{event_id}")
def get_security_event_detail(event_id: str):
    connection = _security_connection()
    try:
        event = get_security_event(connection, event_id)
        if event is None:
            raise HTTPException(status_code=404, detail="安防事件不存在或尚未送达 Agent")
        return event
    finally:
        connection.close()


@app.post("/security/events/{event_id}/actions")
def act_on_security_event(event_id: str, request: SecurityActionRequest):
    operator = request.operator.strip()
    if not operator:
        raise HTTPException(status_code=422, detail="操作人不能为空")
    connection = _security_connection()
    try:
        with connection:
            return handle_security_event(connection, event_id, request.action, operator, request.comment)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    finally:
        connection.close()


@app.get("/security/events/{event_id}/evidence/{evidence_id}")
def get_security_evidence(event_id: str, evidence_id: int):
    connection = _security_connection()
    try:
        path = get_security_evidence_path(connection, event_id, evidence_id)
    finally:
        connection.close()
    if path is None:
        raise HTTPException(status_code=404, detail="安防证据不存在")
    resolved = path.resolve()
    try:
        resolved.relative_to(SAFETY_OUTPUT_ROOT)
    except ValueError as exc:
        raise HTTPException(status_code=403, detail="证据路径不在安全作业目录内") from exc
    if not resolved.is_file():
        raise HTTPException(status_code=404, detail="安防证据文件已丢失")
    return FileResponse(resolved)


if FRONTEND_ROOT.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_ROOT)), name="static")


@app.get("/", include_in_schema=False)
def index():
    index_file = FRONTEND_ROOT / "index.html"
    if not index_file.exists():
        raise HTTPException(
            status_code=503,
            detail="React 前端尚未构建，请先执行 pnpm --dir frontend build",
        )
    return FileResponse(index_file)
