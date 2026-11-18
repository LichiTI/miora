"""
日志查询、单条详情与日志清理路由（/v1/logs、/v1/logs/{log_id}、/v1/logs/cleanup、/v1/backend_logs、/v1/outbound_logs）
"""

from datetime import datetime, timedelta, timezone
from typing import Optional, List, Dict, Any, Literal, Mapping

from fastapi import APIRouter, Depends, HTTPException, Request, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, field_serializer, Field
from sqlalchemy import select, case, func, update, delete, or_

from db import RequestStat, RequestPayload, async_session_scope, DISABLE_DATABASE, DB_TYPE
from routes.deps import rate_limit_dependency, verify_admin_api_key
from core.d1_client import parse_d1_datetime, format_d1_datetime

router = APIRouter()


def _bool_from_db(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        v = value.strip().lower()
        if v in {"1", "true", "yes", "y", "on"}:
            return True
        if v in {"0", "false", "no", "n", "off", ""}:
            return False
    return bool(value)


class LogEntry(BaseModel):
    id: int
    timestamp: datetime
    endpoint: Optional[str] = None
    client_ip: Optional[str] = None
    provider: Optional[str] = None
    model: Optional[str] = None
    api_key_prefix: Optional[str] = None
    process_time: Optional[float] = None
    first_response_time: Optional[float] = None
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    cached_tokens: Optional[int] = None
    cache_creation_tokens: Optional[int] = None
    success: bool = False
    status_code: Optional[int] = None
    prompt_price: Optional[float] = None
    completion_price: Optional[float] = None
    is_flagged: bool = False

    provider_id: Optional[str] = None
    provider_key_index: Optional[int] = None
    api_key_name: Optional[str] = None
    api_key_group: Optional[str] = None
    retry_count: Optional[int] = None
    retry_path: Optional[str] = None
    request_headers: Optional[str] = None
    request_body: Optional[str] = None
    upstream_request_headers: Optional[str] = None
    upstream_request_body: Optional[str] = None
    upstream_response_headers: Optional[str] = None
    upstream_response_body: Optional[str] = None
    response_body: Optional[str] = None
    raw_data_expires_at: Optional[datetime] = None

    @field_serializer("timestamp")
    def serialize_dt(self, dt: datetime):
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.isoformat()

    @field_serializer("raw_data_expires_at")
    def serialize_expires_at(self, dt: Optional[datetime]):
        if dt is None:
            return None
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.isoformat()


class LogsPage(BaseModel):
    items: List[LogEntry]
    total: int
    page: int
    page_size: int
    total_pages: int


LOG_CLEARABLE_FIELDS: Dict[str, str] = {
    "request_headers": "用户请求头(request_headers)",
    "request_body": "用户请求体(request_body)",
    "upstream_request_headers": "上游请求头(upstream_request_headers)",
    "upstream_request_body": "上游请求体(upstream_request_body)",
    "upstream_response_headers": "上游响应头(upstream_response_headers)",
    "upstream_response_body": "上游响应体(upstream_response_body)",
    "response_body": "返回给用户的响应体(response_body)",
    "retry_path": "重试路径(retry_path)",
    "text": "文本摘要(text)",
}

DEFAULT_LOG_CLEANUP_FIELDS: List[str] = [
    "request_headers",
    "request_body",
    "upstream_request_headers",
    "upstream_request_body",
    "upstream_response_headers",
    "upstream_response_body",
    "response_body",
    "retry_path",
]


class LogsCleanupRequest(BaseModel):
    dry_run: bool = True
    action: Literal["clear_fields", "delete_rows"] = "clear_fields"
    fields: List[str] = Field(default_factory=lambda: DEFAULT_LOG_CLEANUP_FIELDS.copy())
    older_than_hours: Optional[int] = Field(default=None, ge=1, le=24 * 3650)
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    provider: Optional[str] = None
    api_key: Optional[str] = None
    model: Optional[str] = None
    success: Optional[bool] = None
    status_codes: Optional[List[int]] = None
    flagged_only: bool = False


class LogsCleanupResponse(BaseModel):
    dry_run: bool
    action: str
    matched_rows: int
    affected_rows: int
    selected_fields: List[str]
    non_null_counts: Dict[str, int]
    filters: Dict[str, Any]
    message: str


LOG_LIST_EXCLUDED_FIELD_NAMES = (
    "request_headers",
    "request_body",
    "upstream_request_headers",
    "upstream_request_body",
    "upstream_response_headers",
    "upstream_response_body",
    "response_body",
)
LOG_DETAIL_FIELD_NAMES = tuple(column.key for column in RequestStat.__table__.columns)
LOG_LIST_COLUMN_NAMES = tuple(
    column_name
    for column_name in LOG_DETAIL_FIELD_NAMES
    if column_name not in LOG_LIST_EXCLUDED_FIELD_NAMES
)
LOG_LIST_SQL_COLUMN_CLAUSE = ", ".join(LOG_LIST_COLUMN_NAMES)


def _normalize_cleanup_fields(fields: List[str]) -> List[str]:
    seen = set()
    normalized: List[str] = []
    for item in fields or []:
        key = str(item or "").strip()
        if not key or key in seen:
            continue
        seen.add(key)
        normalized.append(key)
    return normalized


def parse_datetime_input(dt_input: str) -> datetime:
    """解析 ISO 8601 字符串或 Unix 时间戳"""
    try:
        return datetime.fromtimestamp(float(dt_input), tz=timezone.utc)
    except ValueError:
        try:
            if dt_input.endswith('Z'):
                dt_input = dt_input[:-1] + '+00:00'
            dt_obj = datetime.fromisoformat(dt_input)
            if dt_obj.tzinfo is None:
                dt_obj = dt_obj.replace(tzinfo=timezone.utc)
            return dt_obj.astimezone(timezone.utc)
        except ValueError:
            raise ValueError(
                f"Invalid datetime format: {dt_input}. "
                "Use ISO 8601 (YYYY-MM-DDTHH:MM:SSZ) or Unix timestamp."
            )


def _build_d1_logs_list_sql() -> str:
    return f"SELECT {LOG_LIST_SQL_COLUMN_CLAUSE} FROM request_stats WHERE 1=1"


def _log_list_sa_columns() -> List[Any]:
    return [getattr(RequestStat, column_name) for column_name in LOG_LIST_COLUMN_NAMES]


def _log_detail_sa_columns() -> List[Any]:
    return [getattr(RequestStat, column_name) for column_name in LOG_DETAIL_FIELD_NAMES]


def _to_optional_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _to_optional_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _build_api_key_prefix(raw_api_key: str) -> str:
    if raw_api_key and len(raw_api_key) > 11:
        return f"{raw_api_key[:7]}...{raw_api_key[-4:]}"
    return raw_api_key


def _log_raw_field(row: Mapping[str, Any], field_name: str, *, include_raw_fields: bool, raw_data_expired: bool) -> Optional[str]:
    if not include_raw_fields or raw_data_expired:
        return None
    value = row.get(field_name)
    return str(value) if value is not None else None


def _log_entry_from_mapping(
    row: Mapping[str, Any],
    *,
    include_raw_fields: bool,
    now: Optional[datetime] = None,
) -> LogEntry:
    now = now or datetime.now(timezone.utc)
    timestamp = parse_d1_datetime(row.get("timestamp")) or now
    raw_expires_at = parse_d1_datetime(row.get("raw_data_expires_at"))
    raw_data_expired = raw_expires_at is not None and raw_expires_at < now
    raw_api_key = row.get("api_key") or ""

    return LogEntry(
        id=int(row.get("id") or 0),
        timestamp=timestamp,
        endpoint=row.get("endpoint"),
        client_ip=row.get("client_ip"),
        provider=row.get("provider"),
        model=row.get("model"),
        api_key_prefix=_build_api_key_prefix(str(raw_api_key)),
        process_time=_to_optional_float(row.get("process_time")),
        first_response_time=_to_optional_float(row.get("first_response_time")),
        prompt_tokens=int(row.get("prompt_tokens") or 0),
        completion_tokens=int(row.get("completion_tokens") or 0),
        total_tokens=int(row.get("total_tokens") or 0),
        cached_tokens=int(row.get("cached_tokens") or 0),
        cache_creation_tokens=int(row.get("cache_creation_tokens") or 0),
        success=_bool_from_db(row.get("success")),
        status_code=_to_optional_int(row.get("status_code")),
        prompt_price=_to_optional_float(row.get("prompt_price")),
        completion_price=_to_optional_float(row.get("completion_price")),
        is_flagged=_bool_from_db(row.get("is_flagged")),
        provider_id=row.get("provider_id"),
        provider_key_index=_to_optional_int(row.get("provider_key_index")),
        api_key_name=row.get("api_key_name"),
        api_key_group=row.get("api_key_group"),
        retry_count=_to_optional_int(row.get("retry_count")),
        retry_path=row.get("retry_path") if not raw_data_expired else None,
        request_headers=_log_raw_field(row, "request_headers", include_raw_fields=include_raw_fields, raw_data_expired=raw_data_expired),
        request_body=_log_raw_field(row, "request_body", include_raw_fields=include_raw_fields, raw_data_expired=raw_data_expired),
        upstream_request_headers=_log_raw_field(row, "upstream_request_headers", include_raw_fields=include_raw_fields, raw_data_expired=raw_data_expired),
        upstream_request_body=_log_raw_field(row, "upstream_request_body", include_raw_fields=include_raw_fields, raw_data_expired=raw_data_expired),
        upstream_response_headers=_log_raw_field(row, "upstream_response_headers", include_raw_fields=include_raw_fields, raw_data_expired=raw_data_expired),
        upstream_response_body=_log_raw_field(row, "upstream_response_body", include_raw_fields=include_raw_fields, raw_data_expired=raw_data_expired),
        response_body=_log_raw_field(row, "response_body", include_raw_fields=include_raw_fields, raw_data_expired=raw_data_expired),
        raw_data_expires_at=raw_expires_at,
    )


def _build_cleanup_time_filters(payload: LogsCleanupRequest) -> tuple[Optional[datetime], Optional[datetime], Optional[datetime], Dict[str, Any]]:
    if payload.older_than_hours is not None and (payload.start_time or payload.end_time):
        raise HTTPException(
            status_code=400,
            detail="older_than_hours cannot be used together with start_time/end_time.",
        )

    filters: Dict[str, Any] = {}
    cutoff_dt: Optional[datetime] = None
    if payload.older_than_hours is not None:
        cutoff_dt = datetime.now(timezone.utc) - timedelta(hours=payload.older_than_hours)
        filters["older_than_hours"] = payload.older_than_hours
        filters["older_than_before"] = cutoff_dt.isoformat()

    start_dt: Optional[datetime] = None
    if payload.start_time:
        try:
            start_dt = parse_datetime_input(payload.start_time)
            filters["start_time"] = start_dt.isoformat()
        except ValueError as e:
            raise HTTPException(status_code=400, detail=f"Invalid start_time: {e}") from e

    end_dt: Optional[datetime] = None
    if payload.end_time:
        try:
            end_dt = parse_datetime_input(payload.end_time)
            filters["end_time"] = end_dt.isoformat()
        except ValueError as e:
            raise HTTPException(status_code=400, detail=f"Invalid end_time: {e}") from e

    if start_dt and end_dt and end_dt < start_dt:
        raise HTTPException(status_code=400, detail="end_time must be greater than or equal to start_time.")

    return cutoff_dt, start_dt, end_dt, filters


def _validate_cleanup_request(payload: LogsCleanupRequest) -> tuple[str, List[str]]:
    action = (payload.action or "").strip().lower()
    if action not in {"clear_fields", "delete_rows"}:
        raise HTTPException(status_code=400, detail="Invalid action. Allowed: clear_fields, delete_rows.")

    selected_fields = _normalize_cleanup_fields(payload.fields)
    invalid_fields = [field for field in selected_fields if field not in LOG_CLEARABLE_FIELDS]
    if invalid_fields:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid fields: {invalid_fields}. Allowed fields: {list(LOG_CLEARABLE_FIELDS.keys())}",
        )

    if action == "clear_fields" and not selected_fields:
        raise HTTPException(status_code=400, detail="fields is required when action=clear_fields.")

    if payload.status_codes:
        invalid_codes = [code for code in payload.status_codes if (code < 100 or code > 599)]
        if invalid_codes:
            raise HTTPException(status_code=400, detail=f"Invalid status_codes: {invalid_codes}")

    return action, selected_fields


@router.post("/v1/logs/cleanup", response_model=LogsCleanupResponse, dependencies=[Depends(rate_limit_dependency)])
async def cleanup_logs(
    request: Request,
    payload: LogsCleanupRequest,
    token: str = Depends(verify_admin_api_key),
):
    """按条件清理日志数据（clear_fields 或 delete_rows）。"""
    if DISABLE_DATABASE:
        raise HTTPException(status_code=503, detail="Database is disabled.")
    db_type = (DB_TYPE or "sqlite").lower()

    action, selected_fields = _validate_cleanup_request(payload)
    cutoff_dt, start_dt, end_dt, filters = _build_cleanup_time_filters(payload)

    if payload.provider:
        filters["provider"] = payload.provider
    if payload.api_key:
        filters["api_key"] = payload.api_key
    if payload.model:
        filters["model"] = payload.model
    if payload.success is not None:
        filters["success"] = payload.success
    if payload.status_codes:
        filters["status_codes"] = sorted(set(payload.status_codes))
    if payload.flagged_only:
        filters["flagged_only"] = True

    if db_type == "d1":
        from db import d1_client

        if d1_client is None:
            raise HTTPException(status_code=503, detail="D1 client is not initialized.")

        where_sql_parts: List[str] = ["1=1"]
        params: List[Any] = []

        if cutoff_dt is not None:
            where_sql_parts.append("timestamp < ?")
            params.append(cutoff_dt)
        if start_dt is not None:
            where_sql_parts.append("timestamp >= ?")
            params.append(start_dt)
        if end_dt is not None:
            where_sql_parts.append("timestamp <= ?")
            params.append(end_dt)

        if payload.provider:
            like_value = f"%{payload.provider}%"
            where_sql_parts.append("(provider_id LIKE ? OR provider LIKE ?)")
            params.extend([like_value, like_value])

        if payload.api_key:
            like_value = f"%{payload.api_key}%"
            where_sql_parts.append("(api_key_name LIKE ? OR api_key_group LIKE ? OR api_key LIKE ?)")
            params.extend([like_value, like_value, like_value])

        if payload.model:
            where_sql_parts.append("model LIKE ?")
            params.append(f"%{payload.model}%")

        if payload.success is not None:
            where_sql_parts.append("success = ?")
            params.append(1 if payload.success else 0)

        if payload.status_codes:
            placeholders = ", ".join(["?"] * len(payload.status_codes))
            where_sql_parts.append(f"status_code IN ({placeholders})")
            params.extend(payload.status_codes)

        if payload.flagged_only:
            where_sql_parts.append("is_flagged = 1")

        count_fragments = ["COUNT(*) AS matched_rows"]
        for field in selected_fields:
            if hasattr(RequestPayload, field):
                count_fragments.append(
                    f"SUM(CASE WHEN request_stats.{field} IS NOT NULL OR rp.{field} IS NOT NULL THEN 1 ELSE 0 END) AS {field}"
                )
            else:
                count_fragments.append(f"SUM(CASE WHEN request_stats.{field} IS NOT NULL THEN 1 ELSE 0 END) AS {field}")

        count_sql = (
            f"SELECT {', '.join(count_fragments)} FROM request_stats "
            f"LEFT JOIN request_payloads rp ON rp.request_id = request_stats.request_id "
            f"WHERE {' AND '.join(where_sql_parts)}"
        )
        count_row = await d1_client.query_one(count_sql, params)
        count_row = count_row or {}

        matched_rows = int(count_row.get("matched_rows") or 0)
        non_null_counts = {field: int(count_row.get(field) or 0) for field in selected_fields}

        if payload.dry_run:
            return LogsCleanupResponse(
                dry_run=True,
                action=action,
                matched_rows=matched_rows,
                affected_rows=0,
                selected_fields=selected_fields,
                non_null_counts=non_null_counts,
                filters=filters,
                message="Dry run completed. No changes have been applied.",
            )

        affected_rows = 0
        payload_clear_fields = [f for f in selected_fields if hasattr(RequestPayload, f)]
        if action == "clear_fields":
            set_clause = ", ".join([f"{field} = NULL" for field in selected_fields])
            update_where_parts = list(where_sql_parts)
            if selected_fields:
                update_where_parts.append("(" + " OR ".join([f"{field} IS NOT NULL" for field in selected_fields]) + ")")

            sql = f"UPDATE request_stats SET {set_clause} WHERE {' AND '.join(update_where_parts)}"
            result = await d1_client.execute(sql, params)
            affected_rows = int((result.get("meta") or {}).get("changes") or 0)

            if payload_clear_fields:
                p_set_clause = ", ".join([f"{field} = NULL" for field in payload_clear_fields])
                p_non_null = " OR ".join([f"{field} IS NOT NULL" for field in payload_clear_fields])
                p_sql = (
                    f"UPDATE request_payloads SET {p_set_clause} "
                    f"WHERE request_id IN (SELECT request_id FROM request_stats WHERE {' AND '.join(where_sql_parts)}) "
                    f"AND ({p_non_null})"
                )
                p_res = await d1_client.execute(p_sql, params)
                affected_rows = max(affected_rows, int((p_res.get("meta") or {}).get("changes") or 0))
        else:
            p_del_sql = (
                f"DELETE FROM request_payloads WHERE request_id IN "
                f"(SELECT request_id FROM request_stats WHERE {' AND '.join(where_sql_parts)})"
            )
            await d1_client.execute(p_del_sql, params)
            sql = f"DELETE FROM request_stats WHERE {' AND '.join(where_sql_parts)}"
            result = await d1_client.execute(sql, params)
            affected_rows = int((result.get("meta") or {}).get("changes") or 0)

        return LogsCleanupResponse(
            dry_run=False,
            action=action,
            matched_rows=matched_rows,
            affected_rows=affected_rows,
            selected_fields=selected_fields,
            non_null_counts=non_null_counts,
            filters=filters,
            message="Cleanup applied successfully.",
        )

    conditions = []
    if cutoff_dt is not None:
        conditions.append(RequestStat.timestamp < cutoff_dt)
    if start_dt is not None:
        conditions.append(RequestStat.timestamp >= start_dt)
    if end_dt is not None:
        conditions.append(RequestStat.timestamp <= end_dt)

    if payload.provider:
        conditions.append(
            or_(
                RequestStat.provider_id.ilike(f"%{payload.provider}%"),
                RequestStat.provider.ilike(f"%{payload.provider}%"),
            )
        )

    if payload.api_key:
        conditions.append(
            or_(
                RequestStat.api_key_name.ilike(f"%{payload.api_key}%"),
                RequestStat.api_key_group.ilike(f"%{payload.api_key}%"),
                RequestStat.api_key.ilike(f"%{payload.api_key}%"),
            )
        )

    if payload.model:
        conditions.append(RequestStat.model.ilike(f"%{payload.model}%"))

    if payload.success is not None:
        conditions.append(RequestStat.success == payload.success)

    if payload.status_codes:
        conditions.append(RequestStat.status_code.in_(payload.status_codes))

    if payload.flagged_only:
        conditions.append(RequestStat.is_flagged.is_(True))

    async with async_session_scope() as session:
        aggregate_cols = [func.count(RequestStat.id).label("matched_rows")]
        for field in selected_fields:
            stat_col = getattr(RequestStat, field)
            payload_col = getattr(RequestPayload, field, None)
            if payload_col is not None:
                cond = or_(stat_col.isnot(None), payload_col.isnot(None))
            else:
                cond = stat_col.isnot(None)
            aggregate_cols.append(func.sum(case((cond, 1), else_=0)).label(field))

        count_query = (
            select(*aggregate_cols)
            .select_from(
                RequestStat.__table__.outerjoin(
                    RequestPayload.__table__,
                    RequestPayload.request_id == RequestStat.request_id,
                )
            )
            .where(*conditions)
        )
        count_result = await session.execute(count_query)
        count_row = count_result.mappings().one_or_none() or {}

        matched_rows = int(count_row.get("matched_rows") or 0)
        non_null_counts = {field: int(count_row.get(field) or 0) for field in selected_fields}

        if payload.dry_run:
            return LogsCleanupResponse(
                dry_run=True,
                action=action,
                matched_rows=matched_rows,
                affected_rows=0,
                selected_fields=selected_fields,
                non_null_counts=non_null_counts,
                filters=filters,
                message="Dry run completed. No changes have been applied.",
            )

        matching_req_ids = select(RequestStat.request_id).where(*conditions).where(RequestStat.request_id.isnot(None))
        payload_clear_fields = [f for f in selected_fields if hasattr(RequestPayload, f)]

        if action == "clear_fields":
            values_dict = {field: None for field in selected_fields}
            non_null_clause = or_(*[getattr(RequestStat, field).isnot(None) for field in selected_fields])
            stmt = update(RequestStat).where(*conditions).where(non_null_clause).values(**values_dict)
            exec_result = await session.execute(stmt)
            raw_rowcount = exec_result.rowcount if isinstance(exec_result.rowcount, int) and exec_result.rowcount >= 0 else 0

            p_rowcount = 0
            if payload_clear_fields:
                p_values = {field: None for field in payload_clear_fields}
                p_non_null = or_(*[getattr(RequestPayload, field).isnot(None) for field in payload_clear_fields])
                p_stmt = (
                    update(RequestPayload)
                    .where(RequestPayload.request_id.in_(matching_req_ids))
                    .where(p_non_null)
                    .values(**p_values)
                )
                p_res = await session.execute(p_stmt)
                p_rowcount = p_res.rowcount if isinstance(p_res.rowcount, int) and p_res.rowcount >= 0 else 0

            await session.commit()
            affected_rows = int(max(raw_rowcount, p_rowcount) if (raw_rowcount or p_rowcount) else min(matched_rows, sum(non_null_counts.values())))
        else:
            await session.execute(
                delete(RequestPayload).where(RequestPayload.request_id.in_(matching_req_ids))
            )
            stmt = delete(RequestStat).where(*conditions)
            exec_result = await session.execute(stmt)
            await session.commit()
            raw_rowcount = exec_result.rowcount
            affected_rows = int(raw_rowcount if isinstance(raw_rowcount, int) and raw_rowcount >= 0 else matched_rows)

        return LogsCleanupResponse(
            dry_run=False,
            action=action,
            matched_rows=matched_rows,
            affected_rows=affected_rows,
            selected_fields=selected_fields,
            non_null_counts=non_null_counts,
            filters=filters,
            message="Cleanup applied successfully.",
        )


@router.get("/v1/logs", response_model=LogsPage, dependencies=[Depends(rate_limit_dependency)])
async def get_logs(
    request: Request,
    page: int = Query(1, ge=1, description="Page number (starting from 1)"),
    page_size: int = Query(20, ge=1, le=200, description="Number of items per page"),
    start_time: Optional[str] = Query(None, description="Start time filter (ISO 8601 or Unix timestamp)"),
    end_time: Optional[str] = Query(None, description="End time filter (ISO 8601 or Unix timestamp)"),
    provider: Optional[str] = Query(None, description="Provider/channel filter (fuzzy match)"),
    api_key: Optional[str] = Query(None, description="API key/token filter (fuzzy match)"),
    model: Optional[str] = Query(None, description="Model name filter (fuzzy match)"),
    success: Optional[bool] = Query(None, description="Filter by success status"),
    token: str = Depends(verify_admin_api_key),
):
    """获取请求日志（RequestStat）分页列表，仅管理员可访问。"""
    if DISABLE_DATABASE:
        raise HTTPException(status_code=503, detail="Database is disabled.")
    db_type = (DB_TYPE or "sqlite").lower()

    if db_type == "d1":
        from db import d1_client
        if d1_client is None:
            return LogsPage(items=[], total=0, page=page, page_size=page_size, total_pages=0)

        sql = _build_d1_logs_list_sql()
        params: list[Any] = []

        if start_time:
            try:
                start_dt = parse_datetime_input(start_time)
            except ValueError as e:
                raise HTTPException(status_code=400, detail=f"Invalid start_time: {e}")
            sql += " AND timestamp >= ?"
            params.append(format_d1_datetime(start_dt))

        if end_time:
            try:
                end_dt = parse_datetime_input(end_time)
            except ValueError as e:
                raise HTTPException(status_code=400, detail=f"Invalid end_time: {e}")
            sql += " AND timestamp <= ?"
            params.append(format_d1_datetime(end_dt))

        if provider:
            like_value = f"%{provider}%"
            sql += " AND (provider_id LIKE ? OR provider LIKE ?)"
            params.extend([like_value, like_value])

        if api_key:
            like_value = f"%{api_key}%"
            sql += " AND (api_key_name LIKE ? OR api_key_group LIKE ? OR api_key LIKE ?)"
            params.extend([like_value, like_value, like_value])

        if model:
            like_value = f"%{model}%"
            sql += " AND model LIKE ?"
            params.append(like_value)

        if success is not None:
            success_value = 1 if success else 0
            sql += " AND success = ?"
            params.append(success_value)

        count_sql = sql.replace(f"SELECT {LOG_LIST_SQL_COLUMN_CLAUSE}", "SELECT COUNT(*) AS total", 1)
        total = int(await d1_client.query_value(count_sql, params, column="total", default=0) or 0)
        if total == 0:
            return LogsPage(items=[], total=0, page=page, page_size=page_size, total_pages=0)

        offset = (page - 1) * page_size
        sql += " ORDER BY timestamp DESC LIMIT ? OFFSET ?"
        rows = await d1_client.query_all(sql, [*params, page_size, offset])

        if not rows:
            return LogsPage(items=[], total=0, page=page, page_size=page_size, total_pages=0)
        total_pages = (total + page_size - 1) // page_size if total > 0 else 0
        now = datetime.now(timezone.utc)
        items = [
            _log_entry_from_mapping(row, include_raw_fields=False, now=now)
            for row in rows
        ]

        return LogsPage(
            items=items,
            total=total,
            page=page,
            page_size=page_size,
            total_pages=total_pages,
        )

    async with async_session_scope() as session:
        conditions = []

        if start_time:
            try:
                start_dt = parse_datetime_input(start_time)
                conditions.append(RequestStat.timestamp >= start_dt)
            except ValueError as e:
                raise HTTPException(status_code=400, detail=f"Invalid start_time: {e}")

        if end_time:
            try:
                end_dt = parse_datetime_input(end_time)
                conditions.append(RequestStat.timestamp <= end_dt)
            except ValueError as e:
                raise HTTPException(status_code=400, detail=f"Invalid end_time: {e}")

        if provider:
            conditions.append(
                or_(
                    RequestStat.provider_id.ilike(f"%{provider}%"),
                    RequestStat.provider.ilike(f"%{provider}%")
                )
            )

        if api_key:
            conditions.append(
                or_(
                    RequestStat.api_key_name.ilike(f"%{api_key}%"),
                    RequestStat.api_key_group.ilike(f"%{api_key}%"),
                    RequestStat.api_key.ilike(f"%{api_key}%")
                )
            )

        if model:
            conditions.append(RequestStat.model.ilike(f"%{model}%"))

        if success is not None:
            conditions.append(RequestStat.success == success)

        offset = (page - 1) * page_size

        count_query = select(func.count()).where(*conditions)
        total = (await session.execute(count_query)).scalar() or 0
        if total == 0:
            return LogsPage(items=[], total=0, page=page, page_size=page_size, total_pages=0)

        query = (
            select(*_log_list_sa_columns())
            .where(*conditions)
            .order_by(RequestStat.timestamp.desc())
            .offset(offset)
            .limit(page_size)
        )
        rows_result = await session.execute(query)
        rows = rows_result.mappings().all()

    if not rows:
        return LogsPage(items=[], total=0, page=page, page_size=page_size, total_pages=0)

    total_pages = (total + page_size - 1) // page_size if total > 0 else 0
    now = datetime.now(timezone.utc)
    items = [
        _log_entry_from_mapping(row, include_raw_fields=False, now=now)
        for row in rows
    ]

    return LogsPage(
        items=items,
        total=total,
        page=page,
        page_size=page_size,
        total_pages=total_pages,
    )


@router.get("/v1/logs/{log_id}", response_model=LogEntry, dependencies=[Depends(rate_limit_dependency)])
async def get_log_detail(
    request: Request,
    log_id: int,
    token: str = Depends(verify_admin_api_key),
):
    """获取单条请求日志完整详情，自动合并 request_payloads 冷数据表。"""
    if DISABLE_DATABASE:
        raise HTTPException(status_code=503, detail="Database is disabled.")
    db_type = (DB_TYPE or "sqlite").lower()

    if db_type == "d1":
        from db import d1_client
        if d1_client is None:
            raise HTTPException(status_code=404, detail="Log not found.")

        rows = await d1_client.query_all(
            "SELECT * FROM request_stats WHERE id = ? LIMIT 1",
            [log_id],
        )
        if not rows:
            raise HTTPException(status_code=404, detail="Log not found.")
        merged = dict(rows[0])
        req_id = merged.get("request_id")
        if req_id:
            p_rows = await d1_client.query_all(
                "SELECT * FROM request_payloads WHERE request_id = ? LIMIT 1",
                [req_id],
            )
            if p_rows:
                for k, v in p_rows[0].items():
                    if v is not None:
                        merged[k] = v
        return _log_entry_from_mapping(merged, include_raw_fields=True)

    async with async_session_scope() as session:
        query = (
            select(*_log_detail_sa_columns())
            .where(RequestStat.id == log_id)
            .limit(1)
        )
        result = await session.execute(query)
        row = result.mappings().one_or_none()
        if row is None:
            raise HTTPException(status_code=404, detail="Log not found.")
        merged = dict(row)
        req_id = merged.get("request_id")
        if req_id:
            p_query = select(RequestPayload).where(RequestPayload.request_id == req_id).limit(1)
            p_obj = (await session.execute(p_query)).scalar_one_or_none()
            if p_obj is not None:
                for col in RequestPayload.__table__.columns:
                    val = getattr(p_obj, col.key, None)
                    if val is not None:
                        merged[col.key] = val

    return _log_entry_from_mapping(merged, include_raw_fields=True)


@router.get("/v1/backend_logs", dependencies=[Depends(rate_limit_dependency)])
async def get_backend_logs(
    request: Request,
    since_id: Optional[int] = Query(None, description="Only return entries with id > since_id"),
    limit: int = Query(200, ge=1, le=2000, description="Max entries to return"),
    search: Optional[str] = Query(None, description="Search keyword (case-insensitive)"),
    stream: Optional[str] = Query(None, description="Filter by stream: stdout or stderr"),
    level: Optional[str] = Query(None, description="Filter by exact log level: DEBUG/INFO/WARNING/ERROR/CRITICAL"),
    level_group: Optional[str] = Query(None, description="Filter by level group: errors (ERROR+CRITICAL)"),
    logger_name: Optional[str] = Query(None, description="Filter by logger name (exact, case-insensitive)"),
    token: str = Depends(verify_admin_api_key),
):
    """获取后台进程日志（stdout/stderr 内存缓冲区）。"""
    from core.log_config import get_backend_log_entries

    result = get_backend_log_entries(
        since_id=since_id,
        limit=limit,
        search=search,
        stream=stream,
        level=level,
        level_group=level_group,
        logger_name=logger_name,
    )

    for item in result.get("items", []):
        if hasattr(item.get("captured_at"), "isoformat"):
            item["captured_at"] = item["captured_at"].isoformat()

    return JSONResponse(content=result)


@router.get("/v1/outbound_logs", dependencies=[Depends(rate_limit_dependency)])
async def get_outbound_logs(
    request: Request,
    since_id: Optional[int] = Query(None, description="Only return entries with id > since_id"),
    limit: int = Query(200, ge=1, le=2000, description="Max entries to return"),
    host: Optional[str] = Query(None, description="Filter by target host (fuzzy)"),
    method: Optional[str] = Query(None, description="Filter by HTTP method: GET/POST/..."),
    status_min: Optional[int] = Query(None, description="Min status code (inclusive)"),
    status_max: Optional[int] = Query(None, description="Max status code (inclusive)"),
    search: Optional[str] = Query(None, description="Search keyword in URL (case-insensitive)"),
    token: str = Depends(verify_admin_api_key),
):
    """获取后端出站 HTTP 请求日志（内存缓冲区）。"""
    from core.http import get_outbound_log_entries

    result = get_outbound_log_entries(
        since_id=since_id,
        limit=limit,
        host=host,
        method=method,
        status_min=status_min,
        status_max=status_max,
        search=search,
    )
    return JSONResponse(content=result)
