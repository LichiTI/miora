from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

import core.stats as core_stats
import db
from routes import logs as logs_route


@pytest.mark.asyncio
async def test_batch_write_stats_separates_hot_stats_and_cold_payloads(monkeypatch, tmp_path):
    database_path = tmp_path / "split_stats.db"
    engine = create_async_engine(f"sqlite+aiosqlite:///{database_path.as_posix()}")
    session_factory = async_sessionmaker(engine, expire_on_commit=False)

    @asynccontextmanager
    async def session_scope():
        async with session_factory() as session:
            yield session

    monkeypatch.setattr(core_stats, "DISABLE_DATABASE", False)
    monkeypatch.setattr(core_stats, "DB_TYPE", "sqlite")
    monkeypatch.setattr(core_stats, "async_session_scope", session_scope)
    monkeypatch.setattr(logs_route, "DISABLE_DATABASE", False)
    monkeypatch.setattr(logs_route, "DB_TYPE", "sqlite")
    monkeypatch.setattr(logs_route, "async_session_scope", session_scope)

    async with engine.begin() as conn:
        await conn.run_sync(db.Base.metadata.create_all)

    try:
        expires_at = datetime.now(timezone.utc) + timedelta(hours=24)
        sample_info = {
            "request_id": "req-split-001",
            "endpoint": "POST /v1/chat/completions",
            "client_ip": "127.0.0.1",
            "process_time": 1.25,
            "first_response_time": 0.35,
            "provider": "openai",
            "model": "gpt-4o",
            "api_key": "sk-test-1234567890",
            "success": True,
            "status_code": 200,
            "prompt_tokens": 120,
            "completion_tokens": 80,
            "total_tokens": 200,
            "request_headers": '{"content-type":"application/json"}',
            "request_body": '{"messages":[{"role":"user","content":"hello\\u0000world"}]}'.replace("\\u0000", "\x00"),
            "upstream_request_headers": '{"user-agent":"Miora"}',
            "upstream_request_body": '{"model":"gpt-4o"}',
            "upstream_response_headers": '{"x-request-id":"up-1"}',
            "upstream_response_body": '{"id":"chatcmpl-1"}',
            "response_body": '{"id":"chatcmpl-1","choices":[]}',
            "retry_path": '[{"provider":"openai","status":200}]',
            "raw_data_expires_at": expires_at,
        }

        await core_stats._batch_write_stats([(sample_info, None)])

        async with session_scope() as session:
            stat_row = (await session.execute(select(db.RequestStat))).scalar_one()
            payload_row = (await session.execute(select(db.RequestPayload))).scalar_one()

            assert stat_row.request_id == "req-split-001"
            assert stat_row.request_body is None
            assert stat_row.upstream_request_body is None
            assert stat_row.upstream_response_body is None
            assert stat_row.response_body is None
            assert stat_row.request_headers is None
            assert stat_row.retry_path is None

            assert payload_row.request_id == "req-split-001"
            assert "\x00" not in payload_row.request_body
            assert "helloworld" in payload_row.request_body
            assert payload_row.upstream_request_headers == '{"user-agent":"Miora"}'
            assert payload_row.response_body == '{"id":"chatcmpl-1","choices":[]}'
            assert payload_row.retry_path == '[{"provider":"openai","status":200}]'

            log_id = stat_row.id

        detail = await logs_route.get_log_detail(request=None, log_id=log_id, token="admin")
        assert detail.id == log_id
        assert detail.request_headers == '{"content-type":"application/json"}'
        assert "helloworld" in (detail.request_body or "")
        assert detail.upstream_request_headers == '{"user-agent":"Miora"}'
        assert detail.upstream_response_headers == '{"x-request-id":"up-1"}'
        assert detail.response_body == '{"id":"chatcmpl-1","choices":[]}'
        assert detail.retry_path == '[{"provider":"openai","status":200}]'

        cleanup_req = logs_route.LogsCleanupRequest(
            action="clear_fields",
            fields=["request_body", "response_body"],
            dry_run=False,
        )
        cleanup_res = await logs_route.cleanup_logs(request=None, payload=cleanup_req, token="admin")
        assert cleanup_res.matched_rows == 1
        assert cleanup_res.non_null_counts["request_body"] == 1
        assert cleanup_res.affected_rows == 1

        detail_after = await logs_route.get_log_detail(request=None, log_id=log_id, token="admin")
        assert detail_after.request_body is None
        assert detail_after.response_body is None
        assert detail_after.upstream_request_body == '{"model":"gpt-4o"}'
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_paid_chain_states_batch_deduplicates_queries(monkeypatch):
    calls = []

    async def fake_update(app, key):
        calls.append(key)

    monkeypatch.setattr(core_stats, "update_paid_api_keys_states", fake_update)
    app = SimpleNamespace(
        state=SimpleNamespace(
            paid_api_keys_states={"sk-paid-1": {"credits": 10}},
            config={"api_keys": [{"id": "k1", "api": "sk-paid-1"}]},
        )
    )
    batch = [
        ({"api_key": "sk-paid-1", "api_key_id": "k1", "total_tokens": 100}, app),
        ({"api_key": "sk-paid-1", "api_key_id": "k1", "total_tokens": 200}, app),
        ({"api_key": "sk-paid-1", "api_key_id": "k1", "total_tokens": 300}, app),
    ]
    await core_stats._update_paid_chain_states_batch(batch)
    assert calls == ["sk-paid-1"]
