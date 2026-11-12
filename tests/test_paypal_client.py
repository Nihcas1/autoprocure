"""PayPal client retry behavior against a mocked HTTP transport (no network)."""

from __future__ import annotations

import httpx
import pytest

from paypal_client import PayPalClient, PayPalError

pytestmark = pytest.mark.anyio

TOKEN = httpx.Response(200, json={"access_token": "tok", "expires_in": 3600})
ORDER = {"id": "O1", "status": "COMPLETED", "purchase_units": [{"amount": {"value": "10.00", "currency_code": "USD"}}]}


def client_with(responses: list[httpx.Response]) -> tuple[PayPalClient, list[httpx.Request]]:
    seen: list[httpx.Request] = []
    queue = list(responses)

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/oauth2/token":
            return TOKEN
        seen.append(request)
        return queue.pop(0)

    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return PayPalClient(client_id="id", client_secret="s", base_url="https://paypal.test", http=http,
                        retry_base_seconds=0.0), seen


async def test_429_is_retried_with_same_idempotency_key():
    client, seen = client_with([httpx.Response(429, headers={"Retry-After": "0"}), httpx.Response(200, json=ORDER)])
    order = await client.capture_order("O1", request_id="AP-1-capture")
    assert order.status == "COMPLETED"
    assert len(seen) == 2
    assert {r.headers["PayPal-Request-Id"] for r in seen} == {"AP-1-capture"}


async def test_5xx_is_retried_then_surfaces_status():
    client, seen = client_with([httpx.Response(503, json={}), httpx.Response(503, json={"name": "SERVICE_UNAVAILABLE"})])
    with pytest.raises(PayPalError) as err:
        await client.capture_order("O1", request_id="AP-1-capture")
    assert err.value.status_code == 503 and len(seen) == 2


async def test_4xx_is_not_retried():
    body = {"name": "UNPROCESSABLE_ENTITY", "message": "nope", "details": [{"issue": "INSTRUMENT_DECLINED"}]}
    client, seen = client_with([httpx.Response(422, json=body)])
    with pytest.raises(PayPalError) as err:
        await client.capture_order("O1", request_id="AP-1-capture")
    assert err.value.status_code == 422 and "INSTRUMENT_DECLINED" in str(err.value)
    assert len(seen) == 1


async def test_non_idempotent_call_is_not_retried():
    client, seen = client_with([httpx.Response(503, json={})])
    with pytest.raises(PayPalError):
        await client._request("POST", "/v2/checkout/orders/O1/capture", json={})  # no request id
    assert len(seen) == 1


def test_retry_after_is_capped():
    client, _ = client_with([])
    assert client._backoff(1, "120") == 5.0
