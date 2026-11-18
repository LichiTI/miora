"""
网关运行时契约、请求管道与路由边界回归测试。
"""

import asyncio
import base64
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI, HTTPException

from core.auth import authenticate_admin_api_key_token, verify_admin_api_key
from core.channels.openai_channel import get_gpt_payload
from core.dialects import openai_responses_ws
from core.middleware import StatsMiddleware
from core.models import RequestModel
from core.streaming import LoggingStreamingResponse
from core.utils import scrub_sensitive_error_text
from plugins.file_to_text import MAX_FILE_BASE64_CHARS, _decode_content
from routes import admin as admin_routes
from routes import auth as auth_routes
from routes import setup as setup_routes
from routes import workspace as workspace_routes


def test_client_ip_proxy_hop_selection(monkeypatch):
    monkeypatch.delenv("TRUSTED_PROXIES", raising=False)
    monkeypatch.delenv("TRUST_CF_HEADERS", raising=False)

    # 默认仅信任本机回环反代（127.0.0.1），Docker 网桥/私网 IP (172.18.0.1 / 10.0.0.1) 默认不再盲信转发头
    untrusted_docker_scope = {"client": ("172.18.0.1", 12345)}
    headers = [
        (b"x-forwarded-for", b"1.1.1.1, 198.51.100.42"),
    ]
    assert StatsMiddleware._get_client_ip(untrusted_docker_scope, headers) == "172.18.0.1"

    loopback_scope = {"client": ("127.0.0.1", 12345)}
    assert StatsMiddleware._get_client_ip(loopback_scope, headers) == "198.51.100.42"

    # 默认不盲信 CF-Connecting-IP，防止客户端经过普通 Nginx 反代时伪造 CF-Connecting-IP 绕过限流/黑名单
    headers_with_cf = [
        (b"cf-connecting-ip", b"203.0.113.99"),
        (b"x-forwarded-for", b"1.1.1.1, 198.51.100.42"),
    ]
    assert StatsMiddleware._get_client_ip(loopback_scope, headers_with_cf) == "198.51.100.42"

    # 显式开启 TRUST_CF_HEADERS=true 后才读取 CF-Connecting-IP
    monkeypatch.setenv("TRUST_CF_HEADERS", "true")
    assert StatsMiddleware._get_client_ip(loopback_scope, headers_with_cf) == "203.0.113.99"
    monkeypatch.delenv("TRUST_CF_HEADERS", raising=False)

    ws_mock = SimpleNamespace(
        headers={"x-forwarded-for": "8.8.8.8, 192.0.2.77"},
        client=SimpleNamespace(host="127.0.0.1"),
    )
    assert openai_responses_ws._get_client_ip(ws_mock) == "192.0.2.77"

    direct_scope = {"client": ("198.51.100.200", 54321)}
    assert StatsMiddleware._get_client_ip(direct_scope, headers_with_cf) == "198.51.100.200"


def test_error_text_formatting_redacts_tokens():
    raw_err = (
        "Client error '404 Not Found' for url "
        "'https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-pro:generateContent?key=AIzaSyD1234567890abcdefghijk&alt=sse' "
        "with header Authorization: Bearer sk-proj-supersecretkey1234567890 and custom=my-custom-upstream-secret"
    )
    scrubbed = scrub_sensitive_error_text(raw_err, ["my-custom-upstream-secret"])
    assert "AIzaSyD1234567890abcdefghijk" not in scrubbed
    assert "sk-proj-supersecretkey1234567890" not in scrubbed
    assert "my-custom-upstream-secret" not in scrubbed
    assert "key=***" in scrubbed
    assert "Bearer ***" in scrubbed


def test_admin_key_role_selection():
    conf_only_users = {
        "api_keys": [
            {"api": "zk-user-first", "role": "user"},
            {"api": "zk-user-second", "role": "user"},
        ]
    }
    assert auth_routes._select_admin_api_key_from_config(conf_only_users) is None
    assert setup_routes._select_admin_api_key_from_config(conf_only_users) is None

    app = FastAPI()
    app.state.api_list = ["zk-user-only"]
    app.state.api_keys_db = [{"api": "zk-user-only", "role": "user"}]
    app.state.admin_initialized = True
    app.state.needs_setup = False
    app.state.global_ip_blacklist_matcher = None

    with pytest.raises(HTTPException) as exc_info:
        authenticate_admin_api_key_token(app, "zk-user-only", "127.0.0.1")
    assert exc_info.value.status_code == 403


@pytest.mark.asyncio
async def test_login_attempt_throttle(monkeypatch):
    auth_routes._LOGIN_FAILURES.clear()
    fake_app = FastAPI()
    monkeypatch.setattr(auth_routes, "get_app", lambda: fake_app)
    monkeypatch.setattr(auth_routes, "ensure_initialized", AsyncMock(return_value=None))
    monkeypatch.setattr(
        auth_routes,
        "get_admin_user",
        AsyncMock(return_value=SimpleNamespace(username="admin", password_hash="hashed")),
    )
    monkeypatch.setattr(auth_routes, "verify_password", lambda plain, hashed: plain == "correct-password")

    req = SimpleNamespace(
        headers={"x-forwarded-for": "1.2.3.4, 198.51.100.88"},
        client=SimpleNamespace(host="127.0.0.1"),
    )

    for _ in range(auth_routes.MAX_LOGIN_FAILURES):
        with pytest.raises(HTTPException) as exc_info:
            await auth_routes.login(
                request=req,
                payload=auth_routes.LoginRequest(username="admin", password="wrong-password"),
            )
        assert exc_info.value.status_code == 403

    with pytest.raises(HTTPException) as locked_exc:
        await auth_routes.login(
            request=req,
            payload=auth_routes.LoginRequest(username="admin", password="correct-password"),
        )
    assert locked_exc.value.status_code == 429

    with pytest.raises(HTTPException) as setup_locked_exc:
        await setup_routes.setup_login(
            payload=setup_routes.SetupLoginRequest(username="admin", password="correct-password"),
            request=req,
        )
    assert setup_locked_exc.value.status_code == 429
    auth_routes._LOGIN_FAILURES.clear()


def test_workspace_path_resolution(tmp_path, monkeypatch):
    monkeypatch.setattr(workspace_routes, "PROJECT_ROOT", tmp_path.resolve())
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    secret_file = tmp_path / ".env.production"
    secret_file.write_text("SECRET=123", encoding="utf-8")

    link_path = data_dir / "leak.txt"
    try:
        link_path.symlink_to(secret_file)
    except OSError:
        pytest.skip("当前环境不支持创建测试软链接")

    assert workspace_routes._get_permissions("data/leak.txt") == ""
    with pytest.raises(HTTPException) as exc_info:
        workspace_routes._resolve_path("data/leak.txt")
    assert exc_info.value.status_code == 403


def test_generate_api_key_route_dependency():
    for route in admin_routes.router.routes:
        if getattr(route, "path", None) == "/v1/generate-api-key":
            dep_calls = [dep.call for dep in route.dependant.dependencies]
            assert verify_admin_api_key in dep_calls
            return
    pytest.fail("Route /v1/generate-api-key not found")


def test_file_to_text_base64_limit():
    small_b64 = base64.b64encode(b"hello world").decode("ascii")
    assert _decode_content(small_b64) == "hello world"

    oversized_str = "A" * (MAX_FILE_BASE64_CHARS + 16)
    assert _decode_content(oversized_str) is None


@pytest.mark.asyncio
async def test_openai_stream_usage_and_disconnect_accounting():
    req = RequestModel(
        model="gpt-4o",
        messages=[{"role": "user", "content": "hello"}],
        stream=True,
        stream_options={"include_usage": False},
    )
    provider = {
        "provider": "openai-test",
        "base_url": "https://api.openai.com/v1/chat/completions",
        "model": [{"gpt-4o": "gpt-4o"}],
    }
    _, _, payload = await get_gpt_payload(req, "openai", provider, "sk-test")
    assert payload["stream_options"]["include_usage"] is True

    captured_stats = []

    async def body_gen():
        yield b'data: {"choices":[{"delta":{"content":"Hello world from stream!"}}]}\n\n'
        raise RuntimeError("Simulated client disconnect during send")

    current_info = {
        "start_time": 1000.0,
        "_estimated_prompt_tokens": 25,
    }
    resp = LoggingStreamingResponse(body_gen(), media_type="text/event-stream", current_info=current_info)

    async def fake_send(msg):
        if msg["type"] == "http.response.body" and msg.get("body"):
            return

    from core import streaming as streaming_mod
    orig_enqueue = streaming_mod.enqueue_stats
    streaming_mod.enqueue_stats = lambda info, app=None: captured_stats.append(dict(info))
    try:
        await resp({"type": "http"}, AsyncMock(), fake_send)
    finally:
        streaming_mod.enqueue_stats = orig_enqueue

    assert len(captured_stats) == 1
    recorded = captured_stats[0]
    assert recorded["prompt_tokens"] == 25
    assert recorded["completion_tokens"] >= 1
    assert recorded["total_tokens"] == 25 + recorded["completion_tokens"]


@pytest.mark.asyncio
async def test_model_identifier_validation_and_client_options():
    from core.client_manager import ClientManager
    from core.handler import ModelRequestHandler

    cm = ClientManager()
    await cm.init({})
    async with cm.get_client("https://api.openai.com/v1/chat/completions") as client:
        assert client.follow_redirects is False
    await cm.close()

    app = FastAPI()
    app.state.config = {"api_keys": [{"api": "zk-test", "role": "user", "model": ["*"]}], "providers": []}
    app.state.api_list = ["zk-test"]
    app.state.api_keys_db = app.state.config["api_keys"]
    handler = ModelRequestHandler(app, lambda: {}, AsyncMock())

    for bad_model in ("../v1beta/tunedModels", "gemini-2.5-pro?key=evil", "gpt-4o#frag", "foo\\bar"):
        req = RequestModel(model=bad_model, messages=[{"role": "user", "content": "hi"}])
        with pytest.raises(HTTPException) as exc_info:
            await handler.request_model(req, api_index=0, background_tasks=SimpleNamespace())
        assert exc_info.value.status_code == 400


@pytest.mark.asyncio
async def test_prepare_stream_releases_keepalive_after_45s_for_cloudflare_524(monkeypatch):
    from core import stream_pipeline

    gate = asyncio.Event()

    async def slow_upstream():
        await gate.wait()
        yield 'data: {"choices":[{"delta":{"content":"ok"}}]}\n\n'

    clock = [100.0]

    def fake_monotonic():
        val = clock[0]
        clock[0] += 25.0
        return val

    monkeypatch.setattr(stream_pipeline.time_module, "monotonic", fake_monotonic)

    async def fake_wait_for_timeout(gen, timeout=3, wait_task=None):
        task = wait_task or asyncio.create_task(gen.__anext__())
        return task, "timeout"

    monkeypatch.setattr(stream_pipeline, "wait_for_timeout", fake_wait_for_timeout)

    info = {}
    stream_iter, elapsed = await stream_pipeline.prepare_stream(
        slow_upstream(),
        current_info=info,
        keepalive_interval=15,
        stream_header_timeout=300,
    )
    first_frame = await stream_iter.__anext__()
    assert first_frame == ": keepalive\n\n"
    gate.set()
    await stream_iter.aclose()


@pytest.mark.asyncio
async def test_handler_aborts_retry_when_client_disconnected(monkeypatch):
    from core.handler import ModelRequestHandler
    from core.models import RequestModel

    attempts_made = []

    async def fake_process_request(*args, **kwargs):
        attempts_made.append(1)
        raise HTTPException(status_code=502, detail="upstream error")

    monkeypatch.setattr("core.handler.process_request", fake_process_request)
    monkeypatch.setattr("core.handler.enqueue_stats", lambda *a, **kw: None)

    provider = {
        "provider": "slow-ch",
        "engine": "openai",
        "base_url": "https://api.example.com/v1",
        "api": ["key-1", "key-2", "key-3"],
        "model": ["gpt-4o"],
        "_model_dict_cache": {"gpt-4o": "gpt-4o"},
    }
    app = SimpleNamespace(
        state=SimpleNamespace(
            config={"preferences": {}, "providers": [provider], "api_keys": [{"api": "zk-user", "model": ["all"]}]},
            api_list=["zk-user"],
            provider_timeouts={"global": {"default": 60}},
            keepalive_interval={"global": {"default": 15}},
            error_triggers=[],
            channel_manager=SimpleNamespace(cooldown_period=0),
        )
    )
    handler = ModelRequestHandler(app, lambda: {}, AsyncMock())

    class DisconnectedRawRequest:
        async def is_disconnected(self):
            return True

    req = RequestModel(model="gpt-4o", stream=True, messages=[{"role": "user", "content": "hi"}])
    with pytest.raises(asyncio.CancelledError):
        await handler.request_model(
            req,
            api_index=0,
            background_tasks=SimpleNamespace(),
            override_providers=[provider],
            override_auto_retry=True,
            raw_request=DisconnectedRawRequest(),
        )
    assert len(attempts_made) == 1


@pytest.mark.asyncio
async def test_stats_middleware_rejects_oversized_json_body(monkeypatch):
    import core.middleware as mw

    monkeypatch.setattr(mw, "MAX_REQUEST_BODY_BYTES", 64)
    monkeypatch.setattr(mw, "ensure_initialized", AsyncMock(return_value=None))

    app = FastAPI()
    app.state.config = {"api_keys": [{"api": "zk-test", "model": ["all"]}]}
    app.state.api_list = ["zk-test"]
    app.state.api_keys_db = [{"api": "zk-test", "model": ["all"]}]
    app.state.paid_api_keys_states = {}

    sent_messages = []

    async def dummy_app(scope, receive, send):
        raise AssertionError("Downstream app must not be called when body exceeds limit")

    middleware = StatsMiddleware(dummy_app)

    chunks = [
        {"type": "http.request", "body": b'{"data":"' + b"x" * 100 + b'"}', "more_body": False},
    ]

    async def receive():
        return chunks.pop(0)

    async def send(msg):
        sent_messages.append(msg)

    scope = {
        "type": "http",
        "path": "/v1/chat/completions",
        "method": "POST",
        "headers": [
            (b"content-type", b"application/json"),
            (b"authorization", b"Bearer zk-test"),
        ],
        "client": ("127.0.0.1", 12345),
        "app": app,
    }

    await middleware(scope, receive, send)
    assert any(m.get("type") == "http.response.start" and m.get("status") == 413 for m in sent_messages)


@pytest.mark.asyncio
async def test_csp_headers_and_oauth_nonce():
    import main as main_mod
    from routes.oauth import _oauth_success_page

    assert "frame-src 'none'" in main_mod.DEFAULT_CONTENT_SECURITY_POLICY
    assert "object-src 'none'" in main_mod.DEFAULT_CONTENT_SECURITY_POLICY
    assert "script-src 'self'" in main_mod.DEFAULT_CONTENT_SECURITY_POLICY

    oauth_resp = _oauth_success_page("acct-1", "st-1", "prov-1")
    csp = oauth_resp.headers.get("Content-Security-Policy", "")
    assert "script-src 'nonce-" in csp
    body_text = oauth_resp.body.decode("utf-8")
    assert 'nonce="' in body_text


@pytest.mark.asyncio
async def test_rate_limiter_and_sqlite_path_resolution(monkeypatch):
    import db as db_mod
    from core.rate_limit import InMemoryRateLimiter

    limiter = InMemoryRateLimiter()
    limiter.MAX_KEYS = 4
    for idx in range(6):
        limited = await limiter.is_rate_limited(f"k-{idx}", [(2, 60)])
        assert limited is False
    assert len(limiter.requests) <= 4

    monkeypatch.setattr(db_mod, "DISABLE_DATABASE", False)
    monkeypatch.setattr(db_mod, "DB_TYPE", "sqlite")
    monkeypatch.setattr(db_mod, "DATABASE_URL", "sqlite+aiosqlite:///./custom/test_stats.db")
    assert db_mod.get_sqlite_db_path() == "./custom/test_stats.db"
    await db_mod.init_db()
    assert str(db_mod.db_engine.url) == "sqlite+aiosqlite:///./custom/test_stats.db"
    await db_mod.close_db()


@pytest.mark.asyncio
async def test_oauth_origin_host_matching():
    from routes import oauth as oauth_routes

    class DummyProvider:
        redirect_mode = "auto"

        def build_auth_url(self, state, redirect_uri):
            return f"https://auth.example/authorize?state={state}&redirect_uri={redirect_uri}", "v-1"

    req = SimpleNamespace(
        app=SimpleNamespace(state=SimpleNamespace(oauth_manager=SimpleNamespace(_providers={"ag": DummyProvider()}))),
        headers={"host": "gateway.local:8000"},
        url=SimpleNamespace(scheme="http", netloc="gateway.local:8000"),
        client=SimpleNamespace(host="127.0.0.1"),
    )
    res_evil = await oauth_routes.oauth_authorize(
        type="ag",
        provider="ag-1",
        request=req,
        origin="https://evil.example:9999",
    )
    assert "http://gateway.local:8000/v1/oauth/callback" in res_evil["auth_url"]

    res_same = await oauth_routes.oauth_authorize(
        type="ag",
        provider="ag-1",
        request=req,
        origin="https://gateway.local:8000",
    )
    assert "https://gateway.local:8000/v1/oauth/callback" in res_same["auth_url"]


@pytest.mark.asyncio
async def test_audit_f1_f2_f3_f6_f7_f10_security_hardening():
    import httpx
    import main as main_mod
    from core.auth import _resolve_admin_api_index
    from core.channels.grok_channel import _sso_request_with_safe_redirects, _validate_xai_sso_url
    from core.dialects.router import is_allowed_passthrough_subpath
    from core.jwt_utils import decode_jwt, issue_jwt, revoke_admin_jwts, set_admin_password_fingerprint, set_jwt_secret
    from core.utils import is_valid_model_name
    from routes import oauth as oauth_routes

    # F1: Passthrough subpath allowlist & raw_path validation
    assert is_allowed_passthrough_subpath("claude", "/v1/messages", "/v1/messages/count_tokens", b"/v1/messages/count_tokens")
    assert is_allowed_passthrough_subpath("openai-responses", "/v1/responses", "/v1/responses/compact", b"/v1/responses/compact")
    assert not is_allowed_passthrough_subpath("claude", "/v1/messages", "/v1/messages/batches", b"/v1/messages/batches")
    assert not is_allowed_passthrough_subpath("openai-responses", "/v1/responses", "/v1/responses/resp_123", b"/v1/responses/resp_123")
    assert not is_allowed_passthrough_subpath("claude", "/v1/messages", "/v1/messages/count_tokens", b"/v1/messages/%2e%2e/count_tokens")

    # F2: Grok SSO URL validation & redirect hop check
    assert _validate_xai_sso_url("https://auth.x.ai/oauth2/device/verify").host == "auth.x.ai"
    for bad_sso_url in (
        "http://auth.x.ai/oauth2/device/verify",
        "https://evil.example.com/callback",
        "https://169.254.169.254/latest/meta-data/",
        "https://127.0.0.1/admin",
    ):
        with pytest.raises(ValueError):
            _validate_xai_sso_url(bad_sso_url)

    def _mock_transport(request: httpx.Request) -> httpx.Response:
        if str(request.url) == "https://accounts.x.ai/start":
            return httpx.Response(302, headers={"location": "https://evil.example.com/steal"})
        return httpx.Response(200)

    async with httpx.AsyncClient(transport=httpx.MockTransport(_mock_transport)) as mock_client:
        with pytest.raises(ValueError):
            await _sso_request_with_safe_redirects(mock_client, "GET", "https://accounts.x.ai/start")

    # F3: Admin JWT password fingerprint, revocation, and enabled admin key preference
    set_jwt_secret("audit-test-jwt-secret-key-0123456789")
    set_admin_password_fingerprint("pbkdf2_sha256$260000$salt$hash_v1")
    token_v1 = issue_jwt({"sub": "admin", "role": "admin", "iat": 1000})
    assert decode_jwt(token_v1) is not None

    # Password change invalidates old JWT immediately
    set_admin_password_fingerprint("pbkdf2_sha256$260000$salt$hash_v2")
    assert decode_jwt(token_v1) is None

    token_v2 = issue_jwt({"sub": "admin", "role": "admin", "iat": 2000})
    assert decode_jwt(token_v2) is not None
    revoke_admin_jwts(before_unix=2000)
    assert decode_jwt(token_v2) is None
    set_jwt_secret("trusted-test-secret-reset-after-audit")

    dummy_app = SimpleNamespace(
        state=SimpleNamespace(
            api_keys_db=[
                {"api": "zk-disabled-admin", "role": "admin", "enabled": False},
                {"api": "zk-enabled-admin", "role": "admin", "enabled": True},
            ]
        )
    )
    assert _resolve_admin_api_index(dummy_app) == 1

    # F6: Model name validation rejects multi-segment paths and special delimiters
    assert is_valid_model_name("gemini-2.5-pro")
    assert is_valid_model_name("models/gemini-2.5-pro")
    for invalid_model in (
        "models/gemini-2.5-pro/extra",
        "/gemini-2.5-pro",
        "gemini-2.5-pro/",
        "gemini-2.5-pro;action=evil",
        "gemini-2.5-pro&key=1",
        "gemini 2.5 pro",
    ):
        assert not is_valid_model_name(invalid_model, check_encoded=True)

    # F7: OAuth pending_flows capacity bound
    oauth_routes._pending_flows.clear()
    now_ts = 10000.0
    for idx in range(oauth_routes.MAX_PENDING_OAUTH_FLOWS + 20):
        oauth_routes._pending_flows[f"st-{idx}"] = {"created_at": now_ts + idx}
    oauth_routes._cleanup_expired_flows(now_ts + 50)
    assert len(oauth_routes._pending_flows) <= oauth_routes.MAX_PENDING_OAUTH_FLOWS
    oauth_routes._pending_flows.clear()

    # F10: SPA routes contract
    for required_spa_route in ("/channels", "/playground", "/plugins", "/admin", "/settings", "/logs", "/backend-logs", "/workspace", "/login", "/setup"):
        assert required_spa_route in main_mod.SPA_ROUTES




