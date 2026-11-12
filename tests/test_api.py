"""HTTP layer: FastAPI routes wired to an injected service (no network)."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from agent import UnconfiguredParser
from conftest import FakePayPal, plan
from main import create_app


def test_procure_returns_full_record_with_attempts(make_service):
    svc = make_service(plan(("BC-SCAN-2PK", 1)), plan(("BC-SCAN-WL", 2)), vault_id=None)
    with TestClient(create_app(svc)) as client:
        r = client.post("/procure", json={"request": "two wireless scanners"})
        assert r.status_code == 200
        body = r.json()
        assert len(body["attempts"]) == 2
        assert body["attempts"][0]["policy"]["decision"] == "blocked"
        assert body["policy"]["decision"] == "needs_approval"
        assert body["policy"]["subtotal"] == "178.00"  # money is serialized as an exact string
        assert body["payment"]["approval_url"]
        assert client.get(f"/requests/{body['request_id']}").status_code == 200


def test_paypal_return_captures_after_approval(make_service):
    paypal = FakePayPal()
    svc = make_service(plan(("BC-SCAN-WL", 2)), paypal=paypal)
    with TestClient(create_app(svc)) as client:
        order_id = client.post("/procure", json={"request": "two wireless scanners"}).json()["payment"]["paypal_order_id"]
        paypal.approve(order_id)
        r = client.get("/paypal/return", params={"token": order_id, "PayerID": "X"}, follow_redirects=False)
        assert r.status_code == 303
        request_id = r.headers["location"].split("request=")[1]
        assert client.get(f"/requests/{request_id}").json()["status"] == "paid"
        assert client.get("/policy").json()["budget"]["committed"] == "178.00"


def test_dashboard_is_served(make_service):
    with TestClient(create_app(make_service(plan()))) as client:
        r = client.get("/")
        assert r.status_code == 200 and "AutoProcure" in r.text


def test_unknown_ids_are_404(make_service):
    with TestClient(create_app(make_service(plan()))) as client:
        assert client.get("/requests/AP-NOPE").status_code == 404
        assert client.get("/paypal/return", params={"token": "NOPE"}).status_code == 404


# ------------------------------------------------------------------ API key

KEY = "s3cret-key"
PROTECTED = [
    ("POST", "/procure", {"json": {"request": "two wireless scanners"}}),
    ("GET", "/requests", {}),
    ("POST", "/mandate/setup", {}),
]


@pytest.mark.parametrize("method,path,kwargs", PROTECTED)
@pytest.mark.parametrize("headers", [{}, {"X-API-Key": "wrong"}, {"Authorization": "Bearer wrong"},
                                     {"Authorization": f"Basic {KEY}"}, {"X-API-Key": ""}])
def test_protected_routes_reject_missing_or_wrong_key(make_service, method, path, kwargs, headers):
    svc = make_service(plan(("BC-SCAN-WL", 2)))
    with TestClient(create_app(svc, api_key=KEY)) as client:
        r = client.request(method, path, headers=headers, **kwargs)
        assert r.status_code == 401
        assert r.headers["www-authenticate"] == "Bearer"
    assert not svc.records  # rejected before the agent or PayPal ran


@pytest.mark.parametrize("method,path,kwargs", PROTECTED)
@pytest.mark.parametrize("headers", [{"X-API-Key": KEY}, {"Authorization": f"Bearer {KEY}"},
                                     {"Authorization": f"bearer {KEY}"}])
def test_protected_routes_accept_valid_key(make_service, method, path, kwargs, headers):
    with TestClient(create_app(make_service(plan(("BC-SCAN-WL", 2))), api_key=KEY)) as client:
        assert client.request(method, path, headers=headers, **kwargs).status_code == 200


def test_get_request_by_id_requires_key(make_service):
    with TestClient(create_app(make_service(plan(("BC-SCAN-WL", 2))), api_key=KEY)) as client:
        rid = client.post("/procure", json={"request": "two scanners"}, headers={"X-API-Key": KEY}).json()["request_id"]
        assert client.get(f"/requests/{rid}").status_code == 401
        assert client.get(f"/requests/{rid}", headers={"X-API-Key": KEY}).status_code == 200


def test_public_routes_and_paypal_redirects_work_without_key(make_service):
    paypal = FakePayPal()
    svc = make_service(plan(("BC-SCAN-WL", 2)), paypal=paypal)
    with TestClient(create_app(svc, api_key=KEY)) as client:
        assert client.get("/").status_code == 200
        health = client.get("/health").json()
        assert health["auth_required"] is True
        order_id = client.post("/procure", json={"request": "two scanners"},
                               headers={"X-API-Key": KEY}).json()["payment"]["paypal_order_id"]
        paypal.approve(order_id)
        # PayPal's browser redirect carries no headers and must still capture.
        r = client.get("/paypal/return", params={"token": order_id}, follow_redirects=False)
        assert r.status_code == 303
        assert next(iter(svc.records.values())).status == "paid"


def test_no_key_configured_means_open_access(make_service):
    with TestClient(create_app(make_service(plan(("BC-SCAN-WL", 2))))) as client:
        assert client.get("/health").json()["auth_required"] is False
        assert client.post("/procure", json={"request": "two scanners"}).status_code == 200


def test_blank_api_key_setting_disables_auth():
    from config import Settings
    s = Settings(paypal_client_id="id", paypal_client_secret="secret", autoprocure_api_key="   ",
                 _env_file=None)
    assert s.autoprocure_api_key is None


def test_validation_and_unconfigured_agent(make_service):
    svc = make_service(plan())
    svc.parser = UnconfiguredParser()
    with TestClient(create_app(svc)) as client:
        assert client.post("/procure", json={"request": ""}).status_code == 422
        r = client.post("/procure", json={"request": "two scanners"})
        assert r.status_code == 503 and r.json()["error"] == "agent_not_configured"
        assert client.get("/health").json()["gemini_configured"] is False
