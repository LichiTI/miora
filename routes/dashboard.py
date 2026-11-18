"""
全景遥测仪表盘路由（/v1/stats/dashboard_overview）与多维指标聚合服务
"""

import hashlib
import time as time_module
from datetime import datetime, timedelta, timezone
from typing import Optional, Any, Mapping

from fastapi import APIRouter, Depends, HTTPException, Request, Query
from fastapi.responses import JSONResponse
from sqlalchemy import select, case, func, desc

from db import RequestStat, ChannelStat, async_session_scope, DISABLE_DATABASE, DB_TYPE
from routes.deps import rate_limit_dependency, verify_admin_api_key
from routes.logs import parse_datetime_input, _bool_from_db
from core.d1_client import parse_d1_datetime, format_d1_datetime

router = APIRouter()


def _safe_int(value: Any) -> int:
    if value is None or isinstance(value, bool):
        return 0
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def _safe_float(value: Any) -> float:
    if value is None or isinstance(value, bool):
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _key_analytics_hash(api_key: str) -> str:
    normalized = (api_key or "").strip()
    if not normalized:
        return "anonymous"
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]


def _key_analytics_prefix(api_key: str) -> str:
    normalized = (api_key or "").strip()
    if not normalized:
        return "anonymous"
    if "***" in normalized or "..." in normalized:
        return normalized
    if len(normalized) > 11:
        return f"{normalized[:8]}***"
    return normalized


def _stringify_datetime(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.isoformat()
    parsed = parse_d1_datetime(value)
    if parsed is not None:
        return parsed.isoformat()
    text = str(value).strip()
    return text or None


def _fallback_estimate_model_price(app, model_name: str, provider_name: Optional[str] = None) -> tuple[float, float, float]:
    """解析模型价格（$/M tokens）：优先走渠道/全局/外部价格库，若仍为 0 则按主流模型族提供合理等价估算。"""
    from core.stats import get_current_model_prices
    p_in, p_out, p_cache = get_current_model_prices(app, model_name or "", provider_name=provider_name)
    if p_in > 0 or p_out > 0:
        return p_in, p_out, (p_cache if p_cache > 0 else p_in * 0.1)

    m = (model_name or "").lower()
    if "opus" in m:
        return 15.0, 75.0, 1.5
    if "sonnet" in m:
        return 3.0, 15.0, 0.3
    if "haiku" in m:
        return 0.8, 4.0, 0.08
    if "deepseek" in m:
        if "r1" in m or "reasoner" in m:
            return 0.55, 2.19, 0.14
        return 0.27, 1.10, 0.07
    if "gemini" in m:
        if "flash" in m:
            return 0.30, 2.50, 0.03
        return 1.25, 10.0, 0.125
    if "gpt-4o-mini" in m:
        return 0.15, 0.60, 0.075
    if "gpt-4" in m or "o1" in m or "o3" in m:
        return 2.50, 10.0, 0.25
    return 0.50, 2.00, 0.05


def _new_stat_acc(**extra: Any) -> dict[str, Any]:
    acc: dict[str, Any] = {
        "requests": 0,
        "success_requests": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "cached_tokens": 0,
        "recorded_cost_usd": 0.0,
        "cost_usd": 0.0,
        "sum_pt": 0.0,
        "cnt_pt": 0,
        "sum_frt": 0.0,
        "cnt_frt": 0,
        "gen_comp": 0,
        "gen_dur": 0.0,
    }
    acc.update(extra)
    return acc


def _add_stat_acc(
    acc: dict[str, Any],
    *,
    rc: int,
    ok: int,
    ptok: int,
    ctok: int,
    ttok: int,
    catok: int,
    rec_c: float,
    row_cost: float,
    s_pt: float,
    c_pt: int,
    s_frt: float,
    c_frt: int,
    g_comp: int,
    g_dur: float,
) -> None:
    acc["requests"] += rc
    acc["success_requests"] += ok
    acc["prompt_tokens"] += ptok
    acc["completion_tokens"] += ctok
    acc["total_tokens"] += ttok
    acc["cached_tokens"] += catok
    acc["recorded_cost_usd"] += rec_c
    acc["cost_usd"] += row_cost
    acc["sum_pt"] += s_pt
    acc["cnt_pt"] += c_pt
    acc["sum_frt"] += s_frt
    acc["cnt_frt"] += c_frt
    acc["gen_comp"] += g_comp
    acc["gen_dur"] += g_dur


def _finalize_stat_metrics(
    acc: dict[str, Any],
    *,
    tot_tokens: int = 0,
    tot_cost_usd: float = 0.0,
    rate_digits: int = 1,
) -> dict[str, Any]:
    reqs = acc["requests"]
    oks = acc["success_requests"]
    ptok = acc["prompt_tokens"]
    catok = acc["cached_tokens"]
    spd = (acc["gen_comp"] / acc["gen_dur"]) if acc["gen_dur"] > 0 else 0.0
    return {
        "requests": reqs,
        "success_requests": oks,
        "failed_requests": max(0, reqs - oks),
        "success_rate": round(oks / reqs * 100, rate_digits) if reqs > 0 else 0.0,
        "prompt_tokens": ptok,
        "completion_tokens": acc["completion_tokens"],
        "total_tokens": acc["total_tokens"],
        "cached_tokens": catok,
        "uncached_tokens": max(0, ptok - catok),
        "cache_hit_rate": round(min(100.0, catok / ptok * 100), 1) if ptok > 0 else 0.0,
        "speed_tok_s": round(spd, 1),
        "avg_process_time": round(acc["sum_pt"] / acc["cnt_pt"], 2) if acc["cnt_pt"] > 0 else 0.0,
        "avg_ttft": round(acc["sum_frt"] / acc["cnt_frt"], 2) if acc["cnt_frt"] > 0 else 0.0,
        "recorded_cost_usd": round(acc["recorded_cost_usd"], 4),
        "cost_usd": round(acc["cost_usd"], 4),
        "cost_cny": round(acc["cost_usd"] * 7.25, 2),
        "token_share": round(acc["total_tokens"] / tot_tokens * 100, 2) if tot_tokens > 0 else 0.0,
        "cost_share": round(acc["cost_usd"] / tot_cost_usd * 100, 2) if tot_cost_usd > 0 else 0.0,
    }


def _dashboard_sqlite_queries(has_any_filter: bool) -> dict[str, str]:
    heatmap_group = (
        "COALESCE(model, 'unknown') AS model, COALESCE(provider, 'unknown') AS provider, COALESCE(api_key, '') AS api_key,"
        if has_any_filter
        else ""
    )
    heatmap_group_by = (
        "GROUP BY day, model, provider, COALESCE(api_key, '')"
        if has_any_filter
        else "GROUP BY day"
    )
    return {
        "ts": """
            SELECT
                strftime({p_bucket}, timestamp, {p_tz}) AS bucket,
                COALESCE(model, 'unknown') AS model,
                COALESCE(provider, 'unknown') AS provider,
                COALESCE(api_key, '') AS api_key,
                MAX(api_key_name) AS api_key_name,
                MAX(api_key_group) AS api_key_group,
                MAX(timestamp) AS last_used,
                COUNT(*) AS req_count,
                SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) AS ok_count,
                SUM(COALESCE(prompt_tokens, 0)) AS prompt_tokens,
                SUM(COALESCE(completion_tokens, 0)) AS completion_tokens,
                SUM(COALESCE(total_tokens, 0)) AS total_tokens,
                SUM(COALESCE(cached_tokens, 0)) AS cached_tokens,
                SUM(COALESCE(cache_creation_tokens, 0)) AS cache_creation_tokens,
                SUM((COALESCE(prompt_tokens, 0) * COALESCE(prompt_price, 0.0) + COALESCE(completion_tokens, 0) * COALESCE(completion_price, 0.0)) / 1000000.0) AS recorded_cost,
                SUM(CASE WHEN process_time > 0 THEN process_time ELSE 0 END) AS sum_pt,
                SUM(CASE WHEN process_time > 0 THEN 1 ELSE 0 END) AS cnt_pt,
                SUM(CASE WHEN first_response_time > 0 THEN first_response_time ELSE 0 END) AS sum_frt,
                SUM(CASE WHEN first_response_time > 0 THEN 1 ELSE 0 END) AS cnt_frt,
                SUM(CASE WHEN completion_tokens > 0 AND process_time > 0 THEN completion_tokens ELSE 0 END) AS gen_comp_tokens,
                SUM(CASE WHEN completion_tokens > 0 AND process_time > 0 THEN
                    CASE WHEN first_response_time IS NOT NULL AND (process_time - first_response_time) >= 0.1
                         THEN (process_time - first_response_time)
                         ELSE process_time
                    END
                ELSE 0 END) AS gen_duration
            FROM request_stats
            WHERE timestamp >= {p_start} AND timestamp <= {p_end}
            GROUP BY bucket, model, provider, COALESCE(api_key, '')
            ORDER BY bucket ASC
        """,
        "recent": """
            SELECT id, timestamp, model, provider, api_key, api_key_name, api_key_group,
                   prompt_tokens, completion_tokens, total_tokens, cached_tokens,
                   process_time, first_response_time, success
            FROM request_stats
            WHERE timestamp >= {p_start} AND timestamp <= {p_end}
            ORDER BY timestamp DESC
            LIMIT {p_limit}
        """,
        "heatmap": f"""
            SELECT
                strftime('%Y-%m-%d', timestamp, {{p_tz}}) AS day,
                {heatmap_group}
                COUNT(*) AS req_count,
                SUM(COALESCE(total_tokens, 0)) AS total_tokens
            FROM request_stats
            WHERE timestamp >= {{p_hm_start}}
            {heatmap_group_by}
            ORDER BY day ASC
        """,
    }


_DASHBOARD_HEATMAP_CACHE: dict[tuple[Any, ...], tuple[float, dict[str, Any]]] = {}
_DASHBOARD_HEATMAP_TTL_SEC = 120.0


@router.get("/v1/stats/dashboard_overview", dependencies=[Depends(rate_limit_dependency)])
async def get_dashboard_overview(
    request: Request,
    token: str = Depends(verify_admin_api_key),
    range_key: str = Query(default="24h", description="Time range key: 1h | 6h | 24h | today | 3d | week | month | 30d | all"),
    start_datetime: Optional[str] = Query(default=None, description="Optional custom start ISO datetime"),
    end_datetime: Optional[str] = Query(default=None, description="Optional custom end ISO datetime"),
    model: Optional[str] = Query(default=None, description="Optional comma-separated model filter"),
    provider: Optional[str] = Query(default=None, description="Optional comma-separated provider filter"),
    key_hash: Optional[str] = Query(default=None, description="Optional comma-separated key_hash filter"),
    tz_offset_minutes: int = Query(default=-480, ge=-840, le=840, description="Browser timezoneOffset in minutes (e.g. -480 for UTC+8)"),
    recent_limit: int = Query(default=40, ge=10, le=200),
    include_heatmap: bool = True,
    force_refresh: bool = False,
):
    """仪表盘全景统计端点：单次请求返回全套遥测卡片、多模式时序、渠道成功率、接口分布、最近请求、热力图与多维明细。"""
    if DISABLE_DATABASE:
        return JSONResponse(content={"empty": True})
    db_type = (DB_TYPE or "sqlite").lower()

    app = request.app
    now_utc = datetime.now(timezone.utc)
    local_offset = timedelta(minutes=-tz_offset_minutes)
    local_tz = timezone(local_offset)
    now_local = now_utc.astimezone(local_tz)

    filter_models = {m.strip() for m in (model or "").split(",") if m.strip()}
    filter_providers_raw = {p.strip() for p in (provider or "").split(",") if p.strip()}
    filter_key_hashes = {k.strip().lower() for k in (key_hash or "").split(",") if k.strip()}

    cfg = app.state.config if (app is not None and hasattr(app.state, "config") and isinstance(app.state.config, dict)) else {}
    cfg_keys_list = cfg.get("api_keys") if isinstance(cfg.get("api_keys"), list) else []
    cfg_by_id = {str(item["id"]): item for item in cfg_keys_list if isinstance(item, dict) and item.get("id")}

    cfg_key_meta: dict[str, dict[str, Optional[str]]] = {}
    for item in cfg_keys_list:
        if not isinstance(item, dict):
            continue
        raw_api = str(item.get("api") or "").strip()
        if not raw_api:
            continue
        prefs = item.get("preferences") if isinstance(item.get("preferences"), dict) else {}
        k_name = item.get("name") or prefs.get("name") or None
        k_group = item.get("group") or prefs.get("group") or None
        parent_name: Optional[str] = None
        parent_id = item.get("parent_id")
        if parent_id and str(parent_id) in cfg_by_id:
            p_item = cfg_by_id[str(parent_id)]
            p_prefs = p_item.get("preferences") if isinstance(p_item.get("preferences"), dict) else {}
            parent_name = p_item.get("name") or p_prefs.get("name") or _key_analytics_prefix(str(p_item.get("api") or ""))
        cfg_key_meta[raw_api] = {
            "name": str(k_name) if k_name else None,
            "group": str(k_group) if k_group else None,
            "parent_name": str(parent_name) if parent_name else None,
        }

    key_identity_memo: dict[str, dict[str, Any]] = {}

    def _resolve_key_identity(raw_key_val: Any, db_name: Any = None, db_group: Any = None) -> dict[str, Any]:
        raw_str = str(raw_key_val or "").strip()
        base = key_identity_memo.get(raw_str)
        if base is None:
            if not raw_str:
                base = {
                    "key_hash": "anonymous",
                    "api_key_prefix": "未关联 Key",
                    "cfg_name": None,
                    "cfg_group": None,
                    "parent_name": None,
                    "can_drilldown": False,
                }
            else:
                meta = cfg_key_meta.get(raw_str) or {}
                base = {
                    "key_hash": _key_analytics_hash(raw_str),
                    "api_key_prefix": _key_analytics_prefix(raw_str),
                    "cfg_name": meta.get("name"),
                    "cfg_group": meta.get("group"),
                    "parent_name": meta.get("parent_name"),
                    "can_drilldown": True,
                }
            key_identity_memo[raw_str] = base

        resolved_name = base["cfg_name"] or (str(db_name).strip() if db_name else None) or None
        resolved_group = base["cfg_group"] or (str(db_group).strip() if db_group else None) or None
        return {
            "key_hash": base["key_hash"],
            "key_label": resolved_name or base["api_key_prefix"],
            "api_key_name": resolved_name,
            "api_key_prefix": base["api_key_prefix"],
            "api_key_group": resolved_group,
            "parent_name": base["parent_name"],
            "can_drilldown": base["can_drilldown"],
        }

    def _provider_matches(prov_name: str) -> bool:
        if not filter_providers_raw:
            return True
        return prov_name in filter_providers_raw or any(prov_name.startswith(fp + ":") for fp in filter_providers_raw)

    def _key_matches(k_hash: str) -> bool:
        return not filter_key_hashes or k_hash in filter_key_hashes

    rk = (range_key or "24h").strip().lower()
    end_dt = now_utc
    hours_map = {"1h": 1, "6h": 6, "24h": 24, "3d": 72, "week": 168, "7d": 168, "30d": 720}
    if start_datetime or end_datetime:
        try:
            start_dt: Optional[datetime] = parse_datetime_input(start_datetime) if start_datetime else (now_utc - timedelta(hours=24))
            end_dt = parse_datetime_input(end_datetime) if end_datetime else now_utc
            if start_dt and end_dt < start_dt:
                start_dt, end_dt = end_dt, start_dt
            span_h = max(1.0, (end_dt - start_dt).total_seconds() / 3600.0) if start_dt else 24.0
            granularity = "hour" if span_h <= 96 else "day"
        except ValueError:
            start_dt = now_utc - timedelta(hours=24)
            granularity = "hour"
    elif rk in hours_map:
        h = hours_map[rk]
        start_dt = now_utc - timedelta(hours=h)
        granularity = "hour" if h <= 72 else "day"
    elif rk == "today":
        start_dt = now_local.replace(hour=0, minute=0, second=0, microsecond=0).astimezone(timezone.utc)
        granularity = "hour"
    elif rk == "month":
        start_dt = now_local.replace(day=1, hour=0, minute=0, second=0, microsecond=0).astimezone(timezone.utc)
        granularity = "hour" if (now_utc - start_dt).total_seconds() <= 72 * 3600 else "day"
    elif rk == "all":
        start_dt = None
        granularity = "day"
    else:
        start_dt = now_utc - timedelta(hours=24)
        granularity = "hour"

    heatmap_start_local = (now_local - timedelta(days=364)).replace(hour=0, minute=0, second=0, microsecond=0)
    heatmap_start_dt = heatmap_start_local.astimezone(timezone.utc)
    sqlite_tz_mod = f"{int(local_offset.total_seconds() // 60):+d} minutes"

    price_memo: dict[tuple[str, str], tuple[float, float, float]] = {}

    def _row_cost_usd(prompt_tok: int, comp_tok: int, cached_tok: int, rec_cost: float, model_name: str, provider_name: str) -> float:
        if rec_cost and rec_cost > 0:
            return float(rec_cost)
        key = (model_name or "", provider_name or "")
        prices = price_memo.get(key)
        if prices is None:
            prices = _fallback_estimate_model_price(app, model_name, provider_name)
            price_memo[key] = prices
        p_in, p_out, p_cache = prices
        return (max(0, prompt_tok - cached_tok) * p_in + cached_tok * p_cache + comp_tok * p_out) / 1_000_000.0

    has_any_filter = bool(filter_models or filter_providers_raw or filter_key_hashes)
    fetch_recent_limit = min(800, recent_limit * 5 if has_any_filter else recent_limit)

    def _matches_row(r: Mapping[str, Any]) -> bool:
        if filter_models and str(r.get("model") or "unknown") not in filter_models:
            return False
        if not _provider_matches(str(r.get("provider") or "unknown")):
            return False
        if filter_key_hashes:
            k_ident = _resolve_key_identity(r.get("api_key"), r.get("api_key_name"), r.get("api_key_group"))
            if not _key_matches(k_ident["key_hash"]):
                return False
        return True

    bucket_fmt = "%Y-%m-%d %H:00" if granularity == "hour" else "%Y-%m-%d"
    sql_templates = _dashboard_sqlite_queries(has_any_filter)
    heatmap_rows: list[Mapping[str, Any]] = []
    channel_rows: list[Mapping[str, Any]] = []
    endpoint_rows: list[Mapping[str, Any]] = []

    hm_cache_key = (
        db_type,
        sqlite_tz_mod,
        heatmap_start_local.date().isoformat(),
        tuple(sorted(filter_models)),
        tuple(sorted(filter_providers_raw)),
        tuple(sorted(filter_key_hashes)),
    )
    now_mono = time_module.monotonic()
    cached_hm_entry = _DASHBOARD_HEATMAP_CACHE.get(hm_cache_key) if (include_heatmap and not force_refresh) else None
    cached_heatmap_payload: Optional[dict[str, Any]] = (
        cached_hm_entry[1]
        if (cached_hm_entry is not None and (now_mono - cached_hm_entry[0]) < _DASHBOARD_HEATMAP_TTL_SEC)
        else None
    )
    need_db_heatmap = bool(include_heatmap and cached_heatmap_payload is None)

    if db_type == "d1":
        from db import d1_client
        if d1_client is None:
            raise HTTPException(status_code=503, detail="D1 client is not initialized.")
        if start_dt is None:
            bounds_rows = await d1_client.query_all("SELECT MIN(timestamp) AS min_ts FROM request_stats")
            parsed_min = parse_d1_datetime(bounds_rows[0].get("min_ts")) if bounds_rows and bounds_rows[0].get("min_ts") else None
            if parsed_min:
                start_dt = parsed_min
                if (end_dt - start_dt).total_seconds() <= 96 * 3600:
                    granularity = "hour"
                    bucket_fmt = "%Y-%m-%d %H:00"
        if start_dt is None:
            start_dt = now_utc - timedelta(days=30)

        start_str = format_d1_datetime(start_dt)
        end_str = format_d1_datetime(end_dt)
        heatmap_start_str = format_d1_datetime(heatmap_start_dt)

        fmt_q = {"p_bucket": "?", "p_tz": "?", "p_start": "?", "p_end": "?", "p_limit": "?", "p_hm_start": "?"}
        ts_rows = await d1_client.query_all(sql_templates["ts"].format(**fmt_q), [bucket_fmt, sqlite_tz_mod, start_str, end_str])
        recent_rows_all = await d1_client.query_all(sql_templates["recent"].format(**fmt_q), [start_str, end_str, fetch_recent_limit])
        recent_rows = [r for r in recent_rows_all if _matches_row(r)][:recent_limit]
        if need_db_heatmap:
            heatmap_rows = await d1_client.query_all(sql_templates["heatmap"].format(**fmt_q), [sqlite_tz_mod, heatmap_start_str])
        channel_rows = await d1_client.query_all(
            "SELECT provider, COUNT(*) AS total, SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) AS success_count "
            "FROM channel_stats WHERE timestamp >= ? AND timestamp <= ? GROUP BY provider",
            [start_str, end_str],
        )
        endpoint_rows = await d1_client.query_all(
            "SELECT endpoint, COUNT(*) AS count FROM request_stats "
            "WHERE timestamp >= ? AND timestamp <= ? GROUP BY endpoint ORDER BY count DESC LIMIT 12",
            [start_str, end_str],
        )
    else:
        async with async_session_scope() as session:
            from sqlalchemy import text as sa_text

            if start_dt is None:
                bounds_row = (await session.execute(select(func.min(RequestStat.timestamp).label("min_ts")))).mappings().first()
                parsed_min = parse_d1_datetime(bounds_row.get("min_ts")) if bounds_row and bounds_row.get("min_ts") else None
                if parsed_min:
                    start_dt = parsed_min
                    if (end_dt - start_dt).total_seconds() <= 96 * 3600:
                        granularity = "hour"
                        bucket_fmt = "%Y-%m-%d %H:00"
            if start_dt is None:
                start_dt = now_utc - timedelta(days=30)

            start_str = start_dt.strftime("%Y-%m-%d %H:%M:%S")
            end_str = end_dt.strftime("%Y-%m-%d %H:%M:%S")
            heatmap_start_str = heatmap_start_dt.strftime("%Y-%m-%d %H:%M:%S")

            if db_type == "sqlite":
                fmt_named = {
                    "p_bucket": ":bucket_fmt",
                    "p_tz": ":tz_mod",
                    "p_start": ":start_str",
                    "p_end": ":end_str",
                    "p_limit": ":limit",
                    "p_hm_start": ":heatmap_start_str",
                }
                ts_rows = (await session.execute(sa_text(sql_templates["ts"].format(**fmt_named)), {
                    "bucket_fmt": bucket_fmt,
                    "tz_mod": sqlite_tz_mod,
                    "start_str": start_str,
                    "end_str": end_str,
                })).mappings().all()
                recent_rows_all = (await session.execute(sa_text(sql_templates["recent"].format(**fmt_named)), {
                    "start_str": start_str,
                    "end_str": end_str,
                    "limit": fetch_recent_limit,
                })).mappings().all()
                recent_rows = [r for r in recent_rows_all if _matches_row(r)][:recent_limit]
                if need_db_heatmap:
                    heatmap_rows = (await session.execute(sa_text(sql_templates["heatmap"].format(**fmt_named)), {
                        "tz_mod": sqlite_tz_mod,
                        "heatmap_start_str": heatmap_start_str,
                    })).mappings().all()
                channel_rows = (await session.execute(sa_text(
                    "SELECT provider, COUNT(*) AS total, SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) AS success_count "
                    "FROM channel_stats WHERE timestamp >= :start_str AND timestamp <= :end_str GROUP BY provider"
                ), {"start_str": start_str, "end_str": end_str})).mappings().all()
                endpoint_rows = (await session.execute(sa_text(
                    "SELECT endpoint, COUNT(*) AS count FROM request_stats "
                    "WHERE timestamp >= :start_str AND timestamp <= :end_str GROUP BY endpoint ORDER BY count DESC LIMIT 12"
                ), {"start_str": start_str, "end_str": end_str})).mappings().all()
            else:
                raw_window_rows = (await session.execute(
                    select(
                        RequestStat.id,
                        RequestStat.timestamp,
                        RequestStat.model,
                        RequestStat.provider,
                        RequestStat.api_key,
                        RequestStat.api_key_name,
                        RequestStat.api_key_group,
                        RequestStat.success,
                        RequestStat.status_code,
                        RequestStat.prompt_tokens,
                        RequestStat.completion_tokens,
                        RequestStat.total_tokens,
                        RequestStat.cached_tokens,
                        RequestStat.cache_creation_tokens,
                        RequestStat.prompt_price,
                        RequestStat.completion_price,
                        RequestStat.process_time,
                        RequestStat.first_response_time,
                    )
                    .where(RequestStat.timestamp >= start_dt, RequestStat.timestamp <= end_dt)
                    .order_by(RequestStat.timestamp.desc(), RequestStat.id.desc())
                )).mappings().all()

                ts_agg: dict[tuple[str, str, str, str], dict[str, Any]] = {}
                recent_rows = []
                for r in raw_window_rows:
                    ts_dt = parse_d1_datetime(r.get("timestamp")) or start_dt
                    b_str = ts_dt.astimezone(local_tz).strftime(bucket_fmt)
                    m_str = str(r.get("model") or "unknown")
                    p_str = str(r.get("provider") or "unknown")
                    k_str = str(r.get("api_key") or "")
                    pt = _safe_float(r.get("process_time"))
                    frt = _safe_float(r.get("first_response_time"))
                    ptok = _safe_int(r.get("prompt_tokens"))
                    ctok = _safe_int(r.get("completion_tokens"))
                    ttok = _safe_int(r.get("total_tokens")) or (ptok + ctok)
                    catok = _safe_int(r.get("cached_tokens"))
                    cctok = _safe_int(r.get("cache_creation_tokens"))
                    p_price = _safe_float(r.get("prompt_price"))
                    c_price = _safe_float(r.get("completion_price"))
                    ok_flag = 1 if _bool_from_db(r.get("success")) else 0
                    gen_dur = (pt - frt) if (frt > 0 and (pt - frt) >= 0.1) else pt

                    g_key = (b_str, m_str, p_str, k_str)
                    g = ts_agg.get(g_key)
                    if g is None:
                        g = {
                            "bucket": b_str,
                            "model": m_str,
                            "provider": p_str,
                            "api_key": k_str,
                            "api_key_name": r.get("api_key_name"),
                            "api_key_group": r.get("api_key_group"),
                            "last_used": ts_dt.isoformat(),
                            "req_count": 0,
                            "ok_count": 0,
                            "prompt_tokens": 0,
                            "completion_tokens": 0,
                            "total_tokens": 0,
                            "cached_tokens": 0,
                            "cache_creation_tokens": 0,
                            "recorded_cost": 0.0,
                            "sum_pt": 0.0,
                            "cnt_pt": 0,
                            "sum_frt": 0.0,
                            "cnt_frt": 0,
                            "gen_comp_tokens": 0,
                            "gen_duration": 0.0,
                        }
                        ts_agg[g_key] = g
                    if not g["api_key_name"] and r.get("api_key_name"):
                        g["api_key_name"] = r.get("api_key_name")
                    if not g["api_key_group"] and r.get("api_key_group"):
                        g["api_key_group"] = r.get("api_key_group")
                    g["req_count"] += 1
                    g["ok_count"] += ok_flag
                    g["prompt_tokens"] += ptok
                    g["completion_tokens"] += ctok
                    g["total_tokens"] += ttok
                    g["cached_tokens"] += catok
                    g["cache_creation_tokens"] += cctok
                    g["recorded_cost"] += (ptok * p_price + ctok * c_price) / 1_000_000.0
                    if pt > 0:
                        g["sum_pt"] += pt
                        g["cnt_pt"] += 1
                    if frt > 0:
                        g["sum_frt"] += frt
                        g["cnt_frt"] += 1
                    if ctok > 0 and pt > 0 and gen_dur > 0:
                        g["gen_comp_tokens"] += ctok
                        g["gen_duration"] += gen_dur

                    if len(recent_rows) < recent_limit and _matches_row(r):
                        recent_rows.append(r)

                ts_rows = sorted(ts_agg.values(), key=lambda x: x["bucket"])
                if need_db_heatmap:
                    raw_hm_rows = (await session.execute(
                        select(
                            RequestStat.timestamp,
                            RequestStat.model,
                            RequestStat.provider,
                            RequestStat.api_key,
                            RequestStat.total_tokens,
                        ).where(RequestStat.timestamp >= heatmap_start_dt)
                    )).mappings().all()
                    hm_agg: dict[tuple[str, ...], dict[str, Any]] = {}
                    for hr in raw_hm_rows:
                        ts_dt = parse_d1_datetime(hr.get("timestamp"))
                        if not ts_dt:
                            continue
                        day_str = ts_dt.astimezone(local_tz).strftime("%Y-%m-%d")
                        m_str = str(hr.get("model") or "unknown")
                        p_str = str(hr.get("provider") or "unknown")
                        k_str = str(hr.get("api_key") or "")
                        hk = (day_str, m_str, p_str, k_str) if has_any_filter else (day_str,)
                        he = hm_agg.setdefault(hk, {
                            "day": day_str,
                            "model": m_str,
                            "provider": p_str,
                            "api_key": k_str,
                            "req_count": 0,
                            "total_tokens": 0,
                        })
                        he["req_count"] += 1
                        he["total_tokens"] += _safe_int(hr.get("total_tokens"))
                    heatmap_rows = list(hm_agg.values())

                channel_rows = (await session.execute(
                    select(
                        ChannelStat.provider,
                        func.count().label("total"),
                        func.sum(case((ChannelStat.success, 1), else_=0)).label("success_count"),
                    )
                    .where(ChannelStat.timestamp >= start_dt, ChannelStat.timestamp <= end_dt)
                    .group_by(ChannelStat.provider)
                )).mappings().all()
                endpoint_rows = (await session.execute(
                    select(RequestStat.endpoint, func.count().label("count"))
                    .where(RequestStat.timestamp >= start_dt, RequestStat.timestamp <= end_dt)
                    .group_by(RequestStat.endpoint)
                    .order_by(desc("count"))
                    .limit(12)
                )).mappings().all()

    all_models_meta: dict[str, dict[str, Any]] = {}
    all_providers_meta: dict[str, dict[str, Any]] = {}
    all_keys_meta: dict[str, dict[str, Any]] = {}

    summary_acc = _new_stat_acc(cache_creation_tokens=0)
    bucket_map: dict[str, dict[str, Any]] = {}
    bucket_model_map: dict[str, dict[str, dict[str, int]]] = {}
    bucket_key_map: dict[str, dict[str, dict[str, int]]] = {}
    model_map: dict[str, dict[str, Any]] = {}
    provider_map: dict[str, dict[str, Any]] = {}
    key_map: dict[str, dict[str, Any]] = {}
    cross_map: dict[tuple[str, str], dict[str, Any]] = {}
    key_model_map: dict[tuple[str, str], dict[str, Any]] = {}

    for r in ts_rows:
        b = str(r.get("bucket") or "")
        m = str(r.get("model") or "unknown")
        p = str(r.get("provider") or "unknown")
        k_ident = _resolve_key_identity(r.get("api_key"), r.get("api_key_name"), r.get("api_key_group"))
        k_hash = k_ident["key_hash"]

        rc = _safe_int(r.get("req_count"))
        ok = _safe_int(r.get("ok_count"))
        ptok = _safe_int(r.get("prompt_tokens"))
        ctok = _safe_int(r.get("completion_tokens"))
        ttok = _safe_int(r.get("total_tokens")) or (ptok + ctok)
        catok = _safe_int(r.get("cached_tokens"))
        cctok = _safe_int(r.get("cache_creation_tokens"))
        rec_c = _safe_float(r.get("recorded_cost"))
        row_cost = _row_cost_usd(ptok, ctok, catok, rec_c, m, p)
        s_pt = _safe_float(r.get("sum_pt"))
        c_pt = _safe_int(r.get("cnt_pt"))
        s_frt = _safe_float(r.get("sum_frt"))
        c_frt = _safe_int(r.get("cnt_frt"))
        g_comp = _safe_int(r.get("gen_comp_tokens"))
        g_dur = _safe_float(r.get("gen_duration"))
        row_last_used = _stringify_datetime(r.get("last_used"))

        stat_args = dict(
            rc=rc, ok=ok, ptok=ptok, ctok=ctok, ttok=ttok, catok=catok,
            rec_c=rec_c, row_cost=row_cost, s_pt=s_pt, c_pt=c_pt,
            s_frt=s_frt, c_frt=c_frt, g_comp=g_comp, g_dur=g_dur,
        )

        _add_stat_acc(all_models_meta.setdefault(m, _new_stat_acc(model=m)), **stat_args)
        _add_stat_acc(all_providers_meta.setdefault(p, _new_stat_acc(provider=p)), **stat_args)
        km_all = all_keys_meta.get(k_hash)
        if km_all is None:
            km_all = _new_stat_acc(**k_ident)
            all_keys_meta[k_hash] = km_all
        else:
            if not km_all["api_key_name"] and k_ident["api_key_name"]:
                km_all["api_key_name"] = k_ident["api_key_name"]
                km_all["key_label"] = k_ident["key_label"]
            if not km_all["api_key_group"] and k_ident["api_key_group"]:
                km_all["api_key_group"] = k_ident["api_key_group"]
        _add_stat_acc(km_all, **stat_args)

        if filter_models and m not in filter_models:
            continue
        if not _provider_matches(p) or not _key_matches(k_hash):
            continue

        _add_stat_acc(summary_acc, **stat_args)
        summary_acc["cache_creation_tokens"] += cctok

        if b:
            _add_stat_acc(bucket_map.setdefault(b, _new_stat_acc(bucket=b)), **stat_args)
            bm = bucket_model_map.setdefault(b, {}).setdefault(m, {"requests": 0, "tokens": 0})
            bm["requests"] += rc
            bm["tokens"] += ttok
            bk = bucket_key_map.setdefault(b, {}).setdefault(k_hash, {"requests": 0, "tokens": 0})
            bk["requests"] += rc
            bk["tokens"] += ttok

        _add_stat_acc(model_map.setdefault(m, _new_stat_acc(model=m)), **stat_args)
        _add_stat_acc(provider_map.setdefault(p, _new_stat_acc(provider=p)), **stat_args)
        _add_stat_acc(cross_map.setdefault((p, m), _new_stat_acc(provider=p, model=m)), **stat_args)

        k_entry = key_map.get(k_hash)
        if k_entry is None:
            k_entry = _new_stat_acc(**k_ident, model_tokens={}, last_used=row_last_used)
            key_map[k_hash] = k_entry
        else:
            if not k_entry["api_key_name"] and k_ident["api_key_name"]:
                k_entry["api_key_name"] = k_ident["api_key_name"]
                k_entry["key_label"] = k_ident["key_label"]
            if not k_entry["api_key_group"] and k_ident["api_key_group"]:
                k_entry["api_key_group"] = k_ident["api_key_group"]
            if row_last_used and (not k_entry["last_used"] or row_last_used > k_entry["last_used"]):
                k_entry["last_used"] = row_last_used
        _add_stat_acc(k_entry, **stat_args)
        if m and m != "unknown":
            k_entry["model_tokens"][m] = k_entry["model_tokens"].get(m, 0) + ttok

        km_entry = key_model_map.get((k_hash, m))
        if km_entry is None:
            km_entry = _new_stat_acc(**k_ident, model=m)
            key_model_map[(k_hash, m)] = km_entry
        elif not km_entry["api_key_name"] and k_ident["api_key_name"]:
            km_entry["api_key_name"] = k_ident["api_key_name"]
            km_entry["key_label"] = k_ident["key_label"]
        _add_stat_acc(km_entry, **stat_args)

    available_models = []
    for v in all_models_meta.values():
        if not v["model"] or v["model"] == "unknown":
            continue
        m_fin = _finalize_stat_metrics(v)
        available_models.append({
            "model": v["model"],
            "requests": v["requests"],
            "total_tokens": v["total_tokens"],
            "cache_hit_rate": m_fin["cache_hit_rate"],
            "speed_tok_s": m_fin["speed_tok_s"],
        })
    available_models.sort(key=lambda x: (x["total_tokens"], x["requests"]), reverse=True)

    available_providers = []
    for v in all_providers_meta.values():
        if not v["provider"] or v["provider"] == "unknown":
            continue
        p_fin = _finalize_stat_metrics(v)
        available_providers.append({
            "provider": v["provider"],
            "requests": v["requests"],
            "total_tokens": v["total_tokens"],
            "success_rate": p_fin["success_rate"],
            "cache_hit_rate": p_fin["cache_hit_rate"],
        })
    available_providers.sort(key=lambda x: (x["requests"], x["total_tokens"]), reverse=True)

    available_keys = sorted([
        {
            "key_hash": v["key_hash"],
            "key_label": v["key_label"],
            "api_key_name": v["api_key_name"],
            "api_key_prefix": v["api_key_prefix"],
            "api_key_group": v["api_key_group"],
            "parent_name": v["parent_name"],
            "can_drilldown": v["can_drilldown"],
            **_finalize_stat_metrics(v),
        }
        for v in all_keys_meta.values()
    ], key=lambda x: (x["total_tokens"], x["requests"]), reverse=True)

    timeseries = []
    peak_bucket_speed = 0.0
    bucket_frt_values: list[float] = []
    for b_key in sorted(bucket_map.keys()):
        m_out = _finalize_stat_metrics(bucket_map[b_key], rate_digits=2)
        if m_out["speed_tok_s"] > peak_bucket_speed:
            peak_bucket_speed = m_out["speed_tok_s"]
        if m_out["avg_ttft"] > 0:
            bucket_frt_values.append(m_out["avg_ttft"])
        if m_out["requests"] == 0:
            m_out["success_rate"] = 100.0
        m_out["bucket"] = b_key
        m_out["label"] = b_key[-5:]
        timeseries.append(m_out)

    bucket_frt_values.sort()
    p90_frt = bucket_frt_values[min(len(bucket_frt_values) - 1, int(len(bucket_frt_values) * 0.9))] if bucket_frt_values else 0.0
    avg_speed = (summary_acc["gen_comp"] / summary_acc["gen_dur"]) if summary_acc["gen_dur"] > 0 else 0.0

    tot_tokens = summary_acc["total_tokens"]
    tot_cost_usd = summary_acc["cost_usd"]

    final_models = sorted(
        [{"model": e["model"], **_finalize_stat_metrics(e, tot_tokens=tot_tokens, rate_digits=2)} for e in model_map.values()],
        key=lambda x: (x["total_tokens"], x["requests"]),
        reverse=True,
    )
    final_providers = sorted(
        [{"provider": e["provider"], **_finalize_stat_metrics(e, tot_tokens=tot_tokens, rate_digits=2)} for e in provider_map.values()],
        key=lambda x: (x["total_tokens"], x["requests"]),
        reverse=True,
    )

    key_breakdown = sorted([
        {
            "key_hash": ke["key_hash"],
            "key_label": ke["key_label"],
            "api_key_name": ke["api_key_name"],
            "api_key_prefix": ke["api_key_prefix"],
            "api_key_group": ke["api_key_group"],
            "parent_name": ke["parent_name"],
            "can_drilldown": ke["can_drilldown"],
            **_finalize_stat_metrics(ke, tot_tokens=tot_tokens, tot_cost_usd=tot_cost_usd, rate_digits=2),
            "unique_models": len(ke["model_tokens"]),
            "top_model": max(ke["model_tokens"].items(), key=lambda kv: kv[1])[0] if ke["model_tokens"] else "-",
            "last_used": ke["last_used"],
        }
        for ke in key_map.values()
    ], key=lambda x: (x["total_tokens"], x["requests"]), reverse=True)

    cross_breakdown = sorted(
        [{"provider": ce["provider"], "model": ce["model"], **_finalize_stat_metrics(ce)} for ce in cross_map.values()],
        key=lambda x: (x["total_tokens"], x["requests"]),
        reverse=True,
    )
    key_model_breakdown = sorted([
        {
            "key_hash": kme["key_hash"],
            "key_label": kme["key_label"],
            "api_key_name": kme["api_key_name"],
            "api_key_prefix": kme["api_key_prefix"],
            "api_key_group": kme["api_key_group"],
            "parent_name": kme["parent_name"],
            "can_drilldown": kme["can_drilldown"],
            "model": kme["model"],
            **_finalize_stat_metrics(kme),
        }
        for kme in key_model_map.values()
    ], key=lambda x: (x["total_tokens"], x["requests"]), reverse=True)

    recent_list = []
    for r in recent_rows:
        pt = _safe_float(r.get("process_time"))
        frt = _safe_float(r.get("first_response_time"))
        ctok = _safe_int(r.get("completion_tokens"))
        ptok = _safe_int(r.get("prompt_tokens"))
        ttok = _safe_int(r.get("total_tokens")) or (ptok + ctok)
        gen_d = (pt - frt) if (frt > 0 and (pt - frt) >= 0.1) else pt
        ok_bool = _bool_from_db(r.get("success"))
        k_ident = _resolve_key_identity(r.get("api_key"), r.get("api_key_name"), r.get("api_key_group"))
        recent_list.append({
            "id": _safe_int(r.get("id")),
            "timestamp": _stringify_datetime(r.get("timestamp")),
            "model": r.get("model") or "unknown",
            "provider": r.get("provider") or "unknown",
            "key_hash": k_ident["key_hash"],
            "key_label": k_ident["key_label"],
            "api_key_prefix": k_ident["api_key_prefix"],
            "api_key_group": k_ident["api_key_group"],
            "can_drilldown": k_ident["can_drilldown"],
            "prompt_tokens": ptok,
            "completion_tokens": ctok,
            "total_tokens": ttok,
            "cached_tokens": _safe_int(r.get("cached_tokens")),
            "process_time": round(pt, 2),
            "first_response_time": round(frt, 2),
            "speed_tok_s": round(ctok / gen_d, 1) if (ctok > 0 and gen_d > 0) else 0.0,
            "success": ok_bool,
            "status_code": _safe_int(r.get("status_code")) or (200 if ok_bool else 500),
        })

    activity_heatmap_payload: Optional[dict[str, Any]] = cached_heatmap_payload
    if need_db_heatmap:
        heatmap_by_day: dict[str, dict[str, int]] = {}
        for r in heatmap_rows:
            d_key = str(r.get("day") or "")
            if not d_key:
                continue
            if has_any_filter and not _matches_row(r):
                continue
            slot = heatmap_by_day.setdefault(d_key, {"requests": 0, "total_tokens": 0})
            slot["requests"] += _safe_int(r.get("req_count"))
            slot["total_tokens"] += _safe_int(r.get("total_tokens"))

        heatmap_days = []
        hm_active_days = 0
        hm_total_reqs = 0
        hm_total_tokens = 0
        hm_peak_reqs = 0
        hm_peak_date = ""
        for i in range(365):
            d_str = (heatmap_start_local + timedelta(days=i)).date().isoformat()
            info = heatmap_by_day.get(d_str, {"requests": 0, "total_tokens": 0})
            reqs = info["requests"]
            toks = info["total_tokens"]
            if reqs > 0:
                hm_active_days += 1
                hm_total_reqs += reqs
                hm_total_tokens += toks
                if reqs >= hm_peak_reqs:
                    hm_peak_reqs = reqs
                    hm_peak_date = d_str[5:]
            heatmap_days.append({"date": d_str, "requests": reqs, "total_tokens": toks})
        activity_heatmap_payload = {
            "days": heatmap_days,
            "active_days": hm_active_days,
            "total_requests": hm_total_reqs,
            "total_tokens": hm_total_tokens,
            "peak_day_requests": hm_peak_reqs,
            "peak_day_date": hm_peak_date or "--",
        }
        if len(_DASHBOARD_HEATMAP_CACHE) > 64:
            _DASHBOARD_HEATMAP_CACHE.clear()
        _DASHBOARD_HEATMAP_CACHE[hm_cache_key] = (now_mono, activity_heatmap_payload)

    s_fin = _finalize_stat_metrics(summary_acc, rate_digits=2)
    summary = {
        "range_key": rk,
        "granularity": granularity,
        "start_datetime": start_dt.isoformat(),
        "end_datetime": end_dt.isoformat(),
        "total_tokens": tot_tokens,
        "prompt_tokens": summary_acc["prompt_tokens"],
        "completion_tokens": summary_acc["completion_tokens"],
        "cached_tokens": summary_acc["cached_tokens"],
        "cache_creation_tokens": summary_acc["cache_creation_tokens"],
        "uncached_tokens": s_fin["uncached_tokens"],
        "cache_hit_rate": round(min(100.0, summary_acc["cached_tokens"] / summary_acc["prompt_tokens"] * 100), 2) if summary_acc["prompt_tokens"] > 0 else 0.0,
        "total_requests": summary_acc["requests"],
        "success_requests": summary_acc["success_requests"],
        "failed_requests": s_fin["failed_requests"],
        "success_rate": s_fin["success_rate"],
        "avg_speed_tok_s": round(avg_speed, 1),
        "peak_speed_tok_s": round(peak_bucket_speed, 1),
        "avg_process_time": s_fin["avg_process_time"],
        "avg_first_response_time": s_fin["avg_ttft"],
        "p90_first_response_time": round(p90_frt, 2),
        "total_cost_usd": s_fin["cost_usd"],
        "total_cost_cny": s_fin["cost_cny"],
        "recorded_cost_usd": s_fin["recorded_cost_usd"],
        "active_keys": len(key_breakdown),
    }

    top_models = [m["model"] for m in final_models[:8] if m.get("model")]
    top_keys = [{"key_hash": k["key_hash"], "key_label": k["key_label"]} for k in key_breakdown[:8] if k.get("key_hash")]

    model_timeseries = []
    key_timeseries = []
    for b_key in sorted(bucket_map.keys()):
        short_label = b_key[-5:]
        per_m = bucket_model_map.get(b_key, {})
        row_obj: dict[str, Any] = {"bucket": b_key, "label": short_label}
        for tm in top_models:
            row_obj[tm] = per_m.get(tm, {}).get("requests", 0)
            row_obj[f"{tm}__tokens"] = per_m.get(tm, {}).get("tokens", 0)
        model_timeseries.append(row_obj)

        per_k = bucket_key_map.get(b_key, {})
        k_row_obj: dict[str, Any] = {"bucket": b_key, "label": short_label}
        for tk in top_keys:
            kh = tk["key_hash"]
            k_row_obj[kh] = per_k.get(kh, {}).get("requests", 0)
            k_row_obj[f"{kh}__tokens"] = per_k.get(kh, {}).get("tokens", 0)
        key_timeseries.append(k_row_obj)

    channel_success_rates = sorted(
        [
            {
                "provider": r.get("provider"),
                "success_rate": (_safe_int(r.get("success_count")) / _safe_int(r.get("total"))) if _safe_int(r.get("total")) > 0 else 0.0,
                "total_requests": _safe_int(r.get("total")),
            }
            for r in channel_rows
            if r.get("provider")
        ],
        key=lambda x: x["success_rate"],
        reverse=True,
    )
    endpoint_request_counts = [
        {"endpoint": r.get("endpoint"), "count": _safe_int(r.get("count"))}
        for r in endpoint_rows
        if r.get("endpoint")
    ]

    return JSONResponse(content={
        "summary": summary,
        "timeseries": timeseries,
        "model_timeseries": model_timeseries,
        "key_timeseries": key_timeseries,
        "top_models": top_models,
        "top_keys": top_keys,
        "available_models": available_models,
        "available_providers": available_providers,
        "available_keys": available_keys,
        "recent_requests": recent_list,
        "recent_total": summary_acc["requests"],
        "activity_heatmap": activity_heatmap_payload,
        "model_breakdown": final_models,
        "provider_breakdown": final_providers,
        "key_breakdown": key_breakdown,
        "cross_breakdown": cross_breakdown,
        "key_model_breakdown": key_model_breakdown,
        "channel_success_rates": channel_success_rates,
        "endpoint_request_counts": endpoint_request_counts,
    })
