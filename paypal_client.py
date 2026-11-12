"""Async PayPal REST client: OAuth, Orders v2, Vault v3.

- Access tokens are cached and refreshed 60s before expiry (or on a 401).
- Every mutating call carries a `PayPal-Request-Id`, so retries are idempotent.
- Transient failures (network errors, 5xx) on mutating calls are retried once.
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
from collections.abc import Sequence
from typing import Any

import httpx
from pydantic import BaseModel

from money import fmt
from schemas import PricedLine

log = logging.getLogger("autoprocure.paypal")
MAX_RETRY_SLEEP = 5.0  # seconds; keeps a user-facing request from hanging on a long Retry-After


class PayPalError(Exception):
    def __init__(self, message: str, *, status_code: int | None = None, name: str | None = None,
                 debug_id: str | None = None, details: list[dict[str, Any]] | None = None):
        super().__init__(message)
        self.status_code = status_code
        self.name = name
        self.debug_id = debug_id
        self.details = details or []

    @classmethod
    def from_response(cls, resp: httpx.Response) -> "PayPalError":
        try:
            body = resp.json()
        except ValueError:
            body = {}
        issues = "; ".join(d.get("issue", "") for d in body.get("details", []) if d.get("issue"))
        message = body.get("message") or body.get("error_description") or resp.text[:200] or "PayPal error"
        if issues:
            message = f"{message} ({issues})"
        return cls(message, status_code=resp.status_code, name=body.get("name") or body.get("error"),
                   debug_id=body.get("debug_id"), details=body.get("details"))

    def to_dict(self) -> dict[str, Any]:
        return {"message": str(self), "status_code": self.status_code, "name": self.name,
                "debug_id": self.debug_id, "details": self.details}


class PayPalOrder(BaseModel):
    id: str
    status: str
    amount: str | None = None
    currency: str | None = None
    approval_url: str | None = None
    capture_id: str | None = None
    capture_status: str | None = None

    @classmethod
    def from_api(cls, data: dict[str, Any]) -> "PayPalOrder":
        pu = (data.get("purchase_units") or [{}])[0]
        amount = pu.get("amount") or {}
        captures = (pu.get("payments") or {}).get("captures") or []
        cap = captures[0] if captures else {}
        if not amount and cap.get("amount"):
            amount = cap["amount"]
        approval = next((l["href"] for l in data.get("links", []) if l.get("rel") in ("payer-action", "approve")), None)
        return cls(id=data["id"], status=data.get("status", "UNKNOWN"), amount=amount.get("value"),
                   currency=amount.get("currency_code"), approval_url=approval,
                   capture_id=cap.get("id"), capture_status=cap.get("status"))


class PayPalClient:
    def __init__(self, *, client_id: str, client_secret: str, base_url: str, http: httpx.AsyncClient,
                 brand_name: str = "AutoProcure", retry_base_seconds: float = 0.5):
        self._retry_base = retry_base_seconds
        self._client_id = client_id
        self._client_secret = client_secret
        self._base_url = base_url.rstrip("/")
        self._http = http
        self._brand_name = brand_name
        self._token: str | None = None
        self._token_expiry = 0.0
        self._token_lock = asyncio.Lock()

    # ------------------------------------------------------------------ auth
    async def _get_token(self, *, force: bool = False) -> str:
        async with self._token_lock:
            if not force and self._token and time.monotonic() < self._token_expiry:
                return self._token
            try:
                resp = await self._http.post(
                    f"{self._base_url}/v1/oauth2/token",
                    auth=(self._client_id, self._client_secret),
                    data={"grant_type": "client_credentials"},
                    headers={"Accept": "application/json"},
                )
            except httpx.HTTPError as exc:
                raise PayPalError(f"network error during PayPal auth ({type(exc).__name__})") from exc
            if resp.is_error:
                raise PayPalError.from_response(resp)
            body = resp.json()
            self._token = body["access_token"]
            self._token_expiry = time.monotonic() + max(int(body.get("expires_in", 0)) - 60, 0)
            return self._token

    async def verify_credentials(self) -> None:
        await self._get_token(force=True)

    # --------------------------------------------------------------- request
    def _backoff(self, attempt: int, retry_after: str | None = None) -> float:
        """Honor Retry-After (capped), else exponential backoff with jitter."""
        if retry_after:
            try:
                return min(float(retry_after), MAX_RETRY_SLEEP)
            except ValueError:
                pass  # HTTP-date form; fall through to computed backoff
        return min(self._retry_base * (2 ** (attempt - 1)) + random.uniform(0, self._retry_base), MAX_RETRY_SLEEP)

    async def _request(self, method: str, path: str, *, json: dict[str, Any] | None = None,
                       request_id: str | None = None) -> dict[str, Any]:
        attempts = 2
        refreshed = False
        for attempt in range(1, attempts + 1):
            token = await self._get_token(force=refreshed)
            headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json",
                       "Accept": "application/json", "Prefer": "return=representation"}
            if request_id:
                headers["PayPal-Request-Id"] = request_id
            # Safe to retry only if idempotent (GET, or a PayPal-Request-Id is set).
            can_retry = attempt < attempts and (method == "GET" or bool(request_id))
            try:
                resp = await self._http.request(method, f"{self._base_url}{path}", json=json, headers=headers)
            except httpx.HTTPError as exc:
                if can_retry:
                    log.warning("PayPal %s %s network error, retrying: %r", method, path, exc)
                    await asyncio.sleep(self._backoff(attempt))
                    continue
                raise PayPalError(f"network error calling PayPal ({type(exc).__name__})") from exc

            if resp.status_code == 401 and not refreshed:
                refreshed = True
                continue
            if (resp.status_code == 429 or resp.status_code >= 500) and can_retry:
                log.warning("PayPal %s %s returned %s, retrying", method, path, resp.status_code)
                await asyncio.sleep(self._backoff(attempt, resp.headers.get("Retry-After")))
                continue
            if resp.is_error:
                err = PayPalError.from_response(resp)
                log.error("PayPal %s %s failed: %s (debug_id=%s)", method, path, err, err.debug_id)
                raise err
            return resp.json() if resp.content else {}
        raise PayPalError("PayPal request failed after retries")  # pragma: no cover

    # ---------------------------------------------------------------- orders
    def approval_payment_source(self, *, return_url: str, cancel_url: str) -> dict[str, Any]:
        """Buyer (manager) must approve in the PayPal UI before capture."""
        return {"paypal": {"experience_context": {
            "brand_name": self._brand_name[:127],
            "return_url": return_url,
            "cancel_url": cancel_url,
            "user_action": "PAY_NOW",
            "shipping_preference": "NO_SHIPPING",
        }}}

    @staticmethod
    def vault_payment_source(vault_id: str) -> dict[str, Any]:
        """Merchant-initiated charge against a pre-authorized (vaulted) PayPal wallet."""
        return {"paypal": {"vault_id": vault_id}}

    async def create_order(self, *, lines: Sequence[PricedLine], total: str, currency: str,
                           reference_id: str, description: str, payment_source: dict[str, Any],
                           request_id: str) -> PayPalOrder:
        body = {
            "intent": "CAPTURE",
            "purchase_units": [{
                "reference_id": reference_id[:256],
                "custom_id": reference_id[:127],
                "description": description[:127],
                "amount": {
                    "currency_code": currency,
                    "value": total,
                    "breakdown": {"item_total": {"currency_code": currency, "value": total}},
                },
                "items": [{
                    "name": l.name[:127],
                    "sku": l.sku[:127],
                    "quantity": str(l.quantity),
                    "unit_amount": {"currency_code": currency, "value": fmt(l.unit_price)},
                    "category": "PHYSICAL_GOODS",
                } for l in lines],
            }],
            "payment_source": payment_source,
        }
        data = await self._request("POST", "/v2/checkout/orders", json=body, request_id=request_id)
        return PayPalOrder.from_api(data)

    async def capture_order(self, order_id: str, *, request_id: str) -> PayPalOrder:
        data = await self._request("POST", f"/v2/checkout/orders/{order_id}/capture", json={},
                                   request_id=request_id)
        return PayPalOrder.from_api(data)

    async def get_order(self, order_id: str) -> PayPalOrder:
        return PayPalOrder.from_api(await self._request("GET", f"/v2/checkout/orders/{order_id}"))

    # ----------------------------------------------------------------- vault
    async def create_vault_setup_token(self, *, return_url: str, cancel_url: str,
                                       request_id: str) -> tuple[str, str]:
        """Start a one-time spending authorization. Returns (setup_token_id, approve_url)."""
        body = {"payment_source": {"paypal": {
            "description": f"{self._brand_name} agent spending authorization"[:127],
            "usage_type": "MERCHANT",
            "customer_type": "CONSUMER",
            "permit_multiple_payment_tokens": False,
            "experience_context": {"brand_name": self._brand_name[:127],
                                   "return_url": return_url, "cancel_url": cancel_url},
        }}}
        data = await self._request("POST", "/v3/vault/setup-tokens", json=body, request_id=request_id)
        approve = next((l["href"] for l in data.get("links", []) if l.get("rel") == "approve"), None)
        if not approve:
            raise PayPalError("PayPal did not return an approval link for the setup token")
        return data["id"], approve

    async def create_payment_token(self, setup_token_id: str, *, request_id: str) -> str:
        """Exchange an approved setup token for a reusable vault id."""
        body = {"payment_source": {"token": {"id": setup_token_id, "type": "SETUP_TOKEN"}}}
        data = await self._request("POST", "/v3/vault/payment-tokens", json=body, request_id=request_id)
        return data["id"]
