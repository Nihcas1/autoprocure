"""AutoProcure FastAPI app.

Run:  py -3.14 -m uvicorn main:app --reload --port 8000
"""

from __future__ import annotations

import hmac
import html
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from urllib.parse import quote

import httpx
from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse

from agent import AgentError, AgentNotConfiguredError, GeminiIntentParser, UnconfiguredParser
from catalog import DEFAULT_CATALOG
from config import BASE_DIR, Settings, get_settings
from ledger import SpendLedger
from paypal_client import PayPalClient, PayPalError
from policy import PolicyConfig
from schemas import ProcurementRecord, ProcurementRequest
from service import NotFoundError, ProcurementService

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("autoprocure")
STATIC_DIR = BASE_DIR / "static"


def build_service(settings: Settings, http: httpx.AsyncClient) -> ProcurementService:
    policy = PolicyConfig(
        auto_approve_limit=settings.policy_auto_approve_limit,
        order_hard_cap=settings.policy_order_hard_cap,
        max_unit_price=settings.policy_max_unit_price,
        monthly_budget=settings.policy_monthly_budget,
        max_quantity_per_line=settings.policy_max_quantity_per_line,
        approved_vendors=frozenset(settings.policy_approved_vendors),
    )
    if settings.gemini_api_key:
        parser = GeminiIntentParser(api_key=settings.gemini_api_key.get_secret_value(),
                                    model=settings.gemini_model,
                                    fallback_models=settings.gemini_fallback_models,
                                    approved_vendors=settings.policy_approved_vendors,
                                    max_unit_price=f"{policy.max_unit_price:.2f}",
                                    timeout_seconds=settings.gemini_timeout_seconds)
    else:
        log.warning("GEMINI_API_KEY not set: /procure will return 503 until it is configured.")
        parser = UnconfiguredParser()
    paypal = PayPalClient(client_id=settings.paypal_client_id.get_secret_value(),
                          client_secret=settings.paypal_client_secret.get_secret_value(),
                          base_url=settings.paypal_base_url, http=http, brand_name=settings.brand_name)
    return ProcurementService(parser=parser, catalog=DEFAULT_CATALOG, policy=policy,
                              ledger=SpendLedger(policy.monthly_budget), paypal=paypal,
                              public_base_url=settings.public_base_url, currency=settings.currency,
                              vault_id=settings.paypal_vault_id, max_replans=settings.agent_max_replans)


def _page(title: str, body: str, status_code: int = 200) -> HTMLResponse:
    return HTMLResponse(
        f"<!doctype html><meta charset=utf-8><title>{html.escape(title)}</title>"
        f"<body style='font-family:system-ui;max-width:40rem;margin:3rem auto;padding:0 1rem'>"
        f"<h1>{html.escape(title)}</h1><p>{html.escape(body)}</p></body>",
        status_code=status_code,
    )


def require_api_key(
    request: Request,
    x_api_key: str | None = Header(default=None),
    authorization: str | None = Header(default=None),
) -> None:
    """Accept `X-API-Key: <key>` or `Authorization: Bearer <key>`. No key configured = open."""
    expected: str | None = request.app.state.api_key
    if not expected:
        return
    supplied = x_api_key
    if not supplied and authorization and authorization[:7].lower() == "bearer ":
        supplied = authorization[7:].strip()
    # Constant-time comparison so response timing can't be used to guess the key.
    if not supplied or not hmac.compare_digest(supplied.encode(), expected.encode()):
        raise HTTPException(status_code=401, detail="Missing or invalid API key.",
                            headers={"WWW-Authenticate": "Bearer"})


def create_app(service: ProcurementService | None = None, api_key: str | None = None) -> FastAPI:
    """`service`/`api_key` are injected by tests; otherwise both come from settings at startup."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        if service is not None:
            app.state.service = service
            app.state.api_key = api_key or None
            yield
            return
        settings = get_settings()
        app.state.api_key = (settings.autoprocure_api_key.get_secret_value()
                             if settings.autoprocure_api_key else None)
        if not app.state.api_key:
            log.warning("AUTOPROCURE_API_KEY not set: /procure, /requests and /mandate/setup are OPEN.")
        timeout = httpx.Timeout(settings.paypal_timeout_seconds, connect=5.0, pool=5.0)
        async with httpx.AsyncClient(timeout=timeout) as http:
            app.state.service = build_service(settings, http)
            log.info("AutoProcure up: paypal=%s gemini=%s mandate=%s", settings.paypal_mode,
                     bool(settings.gemini_api_key), bool(settings.paypal_vault_id))
            yield

    app = FastAPI(title="AutoProcure", version="0.1.0", lifespan=lifespan,
                  description="Autonomous, policy-enforced procurement agent on PayPal.")

    def svc(request: Request) -> ProcurementService:
        return request.app.state.service

    # ------------------------------------------------------------ errors
    @app.exception_handler(AgentNotConfiguredError)
    async def _agent_unconfigured(_: Request, exc: AgentNotConfiguredError) -> JSONResponse:
        return JSONResponse(status_code=503, content={"error": "agent_not_configured", "message": str(exc)})

    @app.exception_handler(AgentError)
    async def _agent_error(_: Request, exc: AgentError) -> JSONResponse:
        return JSONResponse(status_code=502, content={"error": "agent_error", "message": str(exc)})

    @app.exception_handler(PayPalError)
    async def _paypal_error(_: Request, exc: PayPalError) -> JSONResponse:
        # Client gets a summary plus PayPal's debug_id (for support); raw details stay in our logs.
        log.error("PayPal error: %s", exc.to_dict())
        return JSONResponse(status_code=502, content={"error": "paypal_error", "message": str(exc),
                                                      "name": exc.name, "debug_id": exc.debug_id})

    @app.middleware("http")
    async def _security_headers(request: Request, call_next):
        response = await call_next(request)
        response.headers.setdefault("X-Frame-Options", "DENY")  # no clickjacking of "Run agent"
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("Referrer-Policy", "no-referrer")  # keep order tokens out of referrers
        return response

    @app.exception_handler(NotFoundError)
    async def _not_found(_: Request, exc: NotFoundError) -> JSONResponse:
        return JSONResponse(status_code=404, content={"error": "not_found", "message": str(exc)})

    # ------------------------------------------------------------ routes
    @app.get("/health")
    async def health(request: Request) -> dict:
        s = svc(request)
        return {"status": "ok", "gemini_configured": not isinstance(s.parser, UnconfiguredParser),
                "spending_mandate_on_file": bool(s.vault_id), "auth_required": bool(request.app.state.api_key)}

    @app.get("/catalog")
    async def catalog(request: Request) -> list[dict]:
        return [i.model_dump(mode="json") for i in svc(request).catalog.all()]

    @app.get("/policy")
    async def policy(request: Request) -> dict:
        s = svc(request)
        cfg = s.policy.model_dump(mode="json")
        cfg["approved_vendors"] = sorted(s.policy.approved_vendors)
        return {"policy": cfg, "budget": s.ledger.snapshot()}

    # --- Protected: these spend money or expose purchase records. ---
    protected = [Depends(require_api_key)]

    @app.post("/procure", response_model=ProcurementRecord, dependencies=protected)
    async def procure(body: ProcurementRequest, request: Request) -> ProcurementRecord:
        return await svc(request).procure(body.request.strip())

    @app.get("/requests", response_model=list[ProcurementRecord], dependencies=protected)
    async def list_requests(request: Request) -> list[ProcurementRecord]:
        return sorted(svc(request).records.values(), key=lambda r: r.created_at, reverse=True)

    @app.get("/requests/{request_id}", response_model=ProcurementRecord, dependencies=protected)
    async def get_request(request_id: str, request: Request) -> ProcurementRecord:
        return svc(request).get(request_id)

    # --- Public: browser redirects from PayPal carry no headers, so they can't require the key.
    # They are safe without it: capture only succeeds for an order the payer already approved
    # in PayPal, and the order id itself comes from PayPal. ---
    # PayPal redirects the approving manager here after they approve / cancel.
    # We capture (or release), then send them back to the dashboard focused on that request.
    @app.get("/paypal/return")
    async def paypal_return(request: Request, token: str = Query(..., description="PayPal order id")) -> RedirectResponse:
        record = await svc(request).complete_approval(token)
        return RedirectResponse(f"/?request={quote(record.request_id)}", status_code=303)

    @app.get("/paypal/cancel")
    async def paypal_cancel(request: Request, token: str = Query(...)) -> RedirectResponse:
        record = await svc(request).cancel_approval(token)
        return RedirectResponse(f"/?request={quote(record.request_id)}", status_code=303)

    @app.get("/", include_in_schema=False)
    async def dashboard() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    # One-time agent spending authorization (PayPal Vault, merchant-initiated payments).
    @app.post("/mandate/setup", dependencies=protected)
    async def mandate_setup(request: Request) -> dict:
        return {"approve_url": await svc(request).start_mandate_setup()}

    @app.get("/mandate/return", response_class=HTMLResponse)
    async def mandate_return(request: Request, approval_token_id: str | None = None,
                             token: str | None = None) -> HTMLResponse:
        setup_token = approval_token_id or token
        if not setup_token:
            return _page("Missing token", "PayPal did not include an approval token.", status_code=400)
        vault_id = await svc(request).complete_mandate_setup(setup_token)
        log.info("Spending mandate created. Add PAYPAL_VAULT_ID=%s to .env to persist it.", vault_id)
        return _page("Agent spending authorized",
                     f"Vault id {vault_id} is active for this session. To keep it across restarts, "
                     f"add PAYPAL_VAULT_ID={vault_id} to .env.")

    @app.get("/mandate/cancel", response_class=HTMLResponse)
    async def mandate_cancel() -> HTMLResponse:
        return _page("Authorization cancelled", "No spending authorization was created.")

    return app


app = create_app()
