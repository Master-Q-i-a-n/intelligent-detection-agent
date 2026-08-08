from __future__ import annotations

import argparse
import hashlib
import re
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional


PROJECT_ROOT = Path(__file__).resolve().parent
RUNTIME_LIBS = PROJECT_ROOT / "runtime_libs"
sys.path.insert(0, str(RUNTIME_LIBS))

import duckdb  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pyarrow as pa  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402


SOURCE_ROOT = Path(r"D:\hf_make\找工作\燃气\数据")
DB_PATH = PROJECT_ROOT / "database" / "gas_ai_input.duckdb"
PARQUET_ROOT = PROJECT_ROOT / "dataset" / "telemetry"
SCHEMA_PATH = PROJECT_ROOT / "schema.sql"


def clean_text(value) -> Optional[str]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return None
    return text


def normalize_id(value) -> Optional[str]:
    text = clean_text(value)
    if text is None:
        return None
    if re.fullmatch(r"\d+\.0", text):
        text = text[:-2]
    return text


def parse_range(value) -> tuple[Optional[float], Optional[float]]:
    text = clean_text(value)
    if not text:
        return None, None
    # 量程中的连字符是区间分隔符，不应被识别成第二个数的负号。
    nums = re.findall(r"\d+(?:\.\d+)?", text.replace("～", "-").replace("—", "-"))
    if len(nums) < 2:
        return None, None
    try:
        low, high = float(nums[0]), float(nums[1])
        return (min(low, high), max(low, high))
    except ValueError:
        return None, None


def stable_id(prefix: str, *parts) -> str:
    raw = "|".join("" if p is None else str(p) for p in parts)
    return f"{prefix}_{hashlib.sha1(raw.encode('utf-8')).hexdigest()[:20]}"


def connect() -> duckdb.DuckDBPyConnection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(DB_PATH))
    con.execute(SCHEMA_PATH.read_text(encoding="utf-8"))
    return con


def replace_table_from_df(con, table: str, df: pd.DataFrame) -> None:
    temp_name = "_incoming_df"
    con.register(temp_name, df)
    cols = [row[1] for row in con.execute(f"PRAGMA table_info('{table}')").fetchall()]
    incoming = [c for c in df.columns if c in cols and c != "loaded_at"]
    con.execute(f"DELETE FROM {table}")
    quoted = ", ".join(f'"{c}"' for c in incoming)
    con.execute(f"INSERT INTO {table} ({quoted}) SELECT {quoted} FROM {temp_name}")
    con.unregister(temp_name)


def build_user_meter(con) -> int:
    path = SOURCE_ROOT / "工商业SCADA清单.xlsx"
    raw = pd.read_excel(path, sheet_name=0, header=1, dtype=object)
    rows = []
    for _, row in raw.iterrows():
        user_id = normalize_id(row.get("用户号"))
        if not user_id:
            continue
        qmin, qmax = parse_range(row.get("量程范围"))
        rows.append(
            {
                "user_id": user_id,
                "station_type": clean_text(row.get("站点类型")),
                "meter_count": pd.to_numeric(row.get("台数"), errors="coerce"),
                "station_name": clean_text(row.get("站点名称")),
                "address": clean_text(row.get("地址")),
                "meter_brand": clean_text(row.get("品牌")),
                "meter_type": clean_text(row.get("类型")),
                "meter_model": clean_text(row.get("仪表型号")),
                "range_text": clean_text(row.get("量程范围")),
                "quantity_min": qmin,
                "quantity_max": qmax,
                "prepaid_flag": clean_text(row.get("是否预付费")),
                "scada_status": clean_text(row.get("SCADA远程数据是否正常")),
                "abnormal_meter_no": clean_text(row.get("异常数据的表号")),
                "installation_status": clean_text(row.get("表是否全部安装完成")),
                "source_file": path.name,
            }
        )
    df = pd.DataFrame(rows).drop_duplicates("user_id", keep="last")
    if not df.empty:
        df["meter_count"] = pd.to_numeric(df["meter_count"], errors="coerce").astype("Int64")
    replace_table_from_df(con, "asset.user_meter", df)
    return len(df)


def build_meter_checks(con) -> tuple[int, int]:
    path = SOURCE_ROOT / "检定数据.xlsx"
    raw = pd.read_excel(path, sheet_name="表具维护检定总表", header=0, dtype=object)
    records = []
    points = []
    for idx, row in raw.iterrows():
        user_id = normalize_id(row.iloc[1] if len(row) > 1 else None)
        meter_no = clean_text(row.iloc[8] if len(row) > 8 else None)
        if not user_id and not meter_no:
            continue
        record_id = stable_id("check", user_id, meter_no, row.iloc[13] if len(row) > 13 else None, idx)
        records.append(
            {
                "check_record_id": record_id,
                "user_id": user_id,
                "company_name": clean_text(row.iloc[2]),
                "meter_brand": clean_text(row.iloc[3]),
                "meter_type": clean_text(row.iloc[4]),
                "meter_model": clean_text(row.iloc[5]),
                "nominal_range": clean_text(row.iloc[6]),
                "diameter": clean_text(row.iloc[7]),
                "base_meter_no": meter_no,
                "corrector_no": clean_text(row.iloc[9]),
                "backup_meter_no": clean_text(row.iloc[10]),
                "offline_time": pd.to_datetime(row.iloc[11], errors="coerce"),
                "online_time": pd.to_datetime(row.iloc[12], errors="coerce"),
                "check_time": pd.to_datetime(row.iloc[13], errors="coerce"),
                "maintenance_status": clean_text(row.iloc[28]),
                "parts_replacement": clean_text(row.iloc[29]),
                "check_status": clean_text(row.iloc[30]),
                "qualification_status": clean_text(row.iloc[31]),
                "linear_data": clean_text(row.iloc[34]),
                "source_row": int(idx + 2),
                "source_file": path.name,
            }
        )
        for point_no in range(1, 5):
            flow = pd.to_numeric(row.iloc[15 + point_no], errors="coerce")
            error = pd.to_numeric(row.iloc[19 + point_no], errors="coerce")
            repeatability = pd.to_numeric(row.iloc[23 + point_no], errors="coerce")
            if pd.isna(flow) and pd.isna(error) and pd.isna(repeatability):
                continue
            points.append(
                {
                    "check_record_id": record_id,
                    "point_no": point_no,
                    "check_flow": None if pd.isna(flow) else float(flow),
                    "indication_error": None if pd.isna(error) else float(error),
                    "repeatability": None if pd.isna(repeatability) else float(repeatability),
                }
            )
    record_df = pd.DataFrame(records)
    point_df = pd.DataFrame(points)
    replace_table_from_df(con, "inspection.meter_check_record", record_df)
    replace_table_from_df(con, "inspection.meter_check_point", point_df)
    return len(record_df), len(point_df)


def build_meter_repairs(con) -> int:
    path = SOURCE_ROOT / "检定数据.xlsx"
    raw = pd.read_excel(path, sheet_name="故障表维修统计表", header=1, dtype=object)
    rows = []
    for idx, row in raw.iterrows():
        company = clean_text(row.iloc[2] if len(row) > 2 else None)
        meter_no = clean_text(row.iloc[7] if len(row) > 7 else None)
        fault = clean_text(row.iloc[8] if len(row) > 8 else None)
        if not any((company, meter_no, fault)):
            continue
        user_id = normalize_id(row.iloc[1] if len(row) > 1 else None)
        rows.append(
            {
                "repair_record_id": stable_id("repair", user_id, meter_no, idx),
                "user_id": user_id,
                "company_name": company,
                "meter_brand": clean_text(row.iloc[3]),
                "meter_type": clean_text(row.iloc[4]),
                "meter_model": clean_text(row.iloc[5]),
                "diameter": clean_text(row.iloc[6]),
                "meter_no": meter_no,
                "fault_description": fault,
                "repair_description": clean_text(row.iloc[9]),
                "offline_time": pd.to_datetime(row.iloc[14], errors="coerce"),
                "online_time": pd.to_datetime(row.iloc[15], errors="coerce"),
                "source_row": int(idx + 3),
                "source_file": path.name,
            }
        )
    df = pd.DataFrame(rows)
    replace_table_from_df(con, "inspection.meter_repair", df)
    return len(df)


def entity_from_filename(path: Path) -> tuple[str, Optional[str], str]:
    stem = path.stem
    match = re.match(r"(?P<id>\d+)-(?P<name>.+)", stem)
    if match:
        user_id = match.group("id")
        name = re.sub(r"^工业用户", "", match.group("name")).strip()
        return f"user:{user_id}", user_id, name
    return f"site:{stem}", None, stem


def read_scada_file(path: Path) -> tuple[pd.DataFrame, int, str, Optional[str]]:
    entity_key, user_id, entity_name = entity_from_filename(path)
    raw = pd.read_excel(path, sheet_name=0, header=0)
    raw.columns = [str(c).strip() for c in raw.columns]
    if "时间戳" not in raw.columns:
        raise ValueError("缺少时间戳列")
    ts = pd.to_datetime(raw["时间戳"], errors="coerce")
    frames = []
    pipeline_count = 0
    for pipe in range(1, 5):
        expected = [
            f"{pipe}号压力",
            f"{pipe}号工况瞬时",
            f"{pipe}号工况累计",
            f"{pipe}号标况瞬时",
            f"{pipe}号标况累计",
            f"{pipe}号温度",
        ]
        if not any(col in raw.columns for col in expected):
            continue
        pipeline_count += 1
        frame = pd.DataFrame(
            {
                "observed_at": ts,
                "entity_key": entity_key,
                "user_id": user_id,
                "entity_name": entity_name,
                "pipeline_no": pipe,
                "pressure": pd.to_numeric(raw.get(f"{pipe}号压力"), errors="coerce"),
                "operational_instant": pd.to_numeric(raw.get(f"{pipe}号工况瞬时"), errors="coerce"),
                "operational_cumulative": pd.to_numeric(raw.get(f"{pipe}号工况累计"), errors="coerce"),
                "standard_instant": pd.to_numeric(raw.get(f"{pipe}号标况瞬时"), errors="coerce"),
                "standard_cumulative": pd.to_numeric(raw.get(f"{pipe}号标况累计"), errors="coerce"),
                "temperature": pd.to_numeric(raw.get(f"{pipe}号温度"), errors="coerce"),
                "source_file": path.name,
            }
        )
        frames.append(frame)
    if not frames:
        raise ValueError("未识别到管路字段")
    result = pd.concat(frames, ignore_index=True)
    result = result[result["observed_at"].notna()].copy()
    result["observation_date"] = result["observed_at"].dt.date
    return result, pipeline_count, entity_key, user_id


def read_scada_file_safe(path: Path):
    try:
        frame, pipeline_count, entity_key, user_id = read_scada_file(path)
        return True, frame, pipeline_count, entity_key, user_id, None
    except Exception as exc:
        entity_key, user_id, _ = entity_from_filename(path)
        return False, None, 0, entity_key, user_id, str(exc)[:500]


def iter_chunks(items: list[Path], size: int) -> Iterable[list[Path]]:
    for pos in range(0, len(items), size):
        yield items[pos : pos + size]


def build_scada(con, overwrite: bool, limit_files: Optional[int], chunk_size: int, workers: int) -> dict:
    date_dirs = sorted(p for p in SOURCE_ROOT.iterdir() if p.is_dir() and re.fullmatch(r"\d{4}-\d{2}-\d{2}", p.name))
    if limit_files:
        all_paths = [p for d in date_dirs for p in sorted(d.glob("*.xls"))][:limit_files]
        grouped = {}
        for path in all_paths:
            grouped.setdefault(path.parent.name, []).append(path)
        date_items = sorted(grouped.items())
    else:
        date_items = [(d.name, sorted(d.glob("*.xls"))) for d in date_dirs]

    if overwrite and PARQUET_ROOT.exists():
        resolved = PARQUET_ROOT.resolve()
        if PROJECT_ROOT.resolve() not in resolved.parents:
            raise RuntimeError("拒绝删除项目目录之外的路径")
        shutil.rmtree(resolved)
        con.execute("DELETE FROM telemetry.import_file_log")
    PARQUET_ROOT.mkdir(parents=True, exist_ok=True)

    total_files = 0
    total_rows = 0
    failed = 0
    executor = ProcessPoolExecutor(max_workers=max(1, workers)) if workers > 1 else None
    try:
        for date_name, paths in date_items:
            partition_dir = PARQUET_ROOT / f"observation_date={date_name}"
            partition_dir.mkdir(parents=True, exist_ok=True)
            for part_no, chunk in enumerate(iter_chunks(paths, chunk_size)):
                frames = []
                logs = []
                results = list(executor.map(read_scada_file_safe, chunk)) if executor else [read_scada_file_safe(p) for p in chunk]
                for path, result in zip(chunk, results):
                    ok, frame, pipe_count, entity_key, user_id, error_message = result
                    if ok:
                        assert frame is not None
                        frames.append(frame.drop(columns=["observation_date"]))
                        logs.append(
                            {
                                "source_file": str(path.relative_to(SOURCE_ROOT)),
                                "observation_date": pd.to_datetime(date_name).date(),
                                "entity_key": entity_key,
                                "user_id": user_id,
                                "row_count": len(frame),
                                "pipeline_count": pipe_count,
                                "status": "success",
                                "error_message": None,
                            }
                        )
                        total_files += 1
                        total_rows += len(frame)
                    else:
                        failed += 1
                        logs.append(
                            {
                                "source_file": str(path.relative_to(SOURCE_ROOT)),
                                "observation_date": pd.to_datetime(date_name).date(),
                                "entity_key": entity_key,
                                "user_id": user_id,
                                "row_count": 0,
                                "pipeline_count": 0,
                                "status": "failed",
                                "error_message": error_message,
                            }
                        )
                if frames:
                    batch = pd.concat(frames, ignore_index=True)
                    table = pa.Table.from_pandas(batch, preserve_index=False)
                    pq.write_table(table, partition_dir / f"part-{part_no:05d}.parquet", compression="zstd")
                if logs:
                    log_df = pd.DataFrame(logs)
                    con.register("_logs", log_df)
                    con.execute("DELETE FROM telemetry.import_file_log WHERE source_file IN (SELECT source_file FROM _logs)")
                    con.execute(
                        """
                        INSERT INTO telemetry.import_file_log
                        (source_file, observation_date, entity_key, user_id, row_count, pipeline_count, status, error_message)
                        SELECT source_file, observation_date, entity_key, user_id, row_count, pipeline_count, status, error_message
                        FROM _logs
                        """
                    )
                    con.unregister("_logs")
            print(f"[{datetime.now():%H:%M:%S}] {date_name}: {len(paths)} files processed", flush=True)
    finally:
        if executor:
            executor.shutdown(wait=True)

    parquet_glob = (PARQUET_ROOT / "observation_date=*" / "*.parquet").as_posix()
    con.execute("DROP VIEW IF EXISTS telemetry.scada_observation")
    con.execute(
        f"""
        CREATE VIEW telemetry.scada_observation AS
        SELECT *, CAST(observation_date AS DATE) AS data_date
        FROM read_parquet('{parquet_glob}', hive_partitioning=1)
        """
    )
    return {"files": total_files, "rows": total_rows, "failed": failed, "dates": len(date_items)}


def build_master(con) -> dict:
    users = build_user_meter(con)
    checks, points = build_meter_checks(con)
    repairs = build_meter_repairs(con)
    return {"users": users, "checks": checks, "check_points": points, "repairs": repairs}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build three logical input libraries for the gas AI agent")
    parser.add_argument("--mode", choices=["master", "scada", "all"], default="all")
    parser.add_argument("--overwrite", action="store_true", help="Rebuild generated telemetry parquet files")
    parser.add_argument("--limit-files", type=int, default=None, help="Only process the first N XLS files for validation")
    parser.add_argument("--chunk-size", type=int, default=40)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    con = connect()
    try:
        if args.mode in ("master", "all"):
            print("MASTER", build_master(con), flush=True)
        if args.mode in ("scada", "all"):
            print("SCADA", build_scada(con, args.overwrite, args.limit_files, args.chunk_size, args.workers), flush=True)
    finally:
        con.close()


if __name__ == "__main__":
    main()
