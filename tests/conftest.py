"""Shared fixtures: a scripted planner and a fake PayPal, so tests are offline and deterministic."""

from __future__ import annotations

import sys
from decimal import Decimal
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from catalog import DEFAULT_CATALOG, Catalog  # noqa: E402
from ledger import SpendLedger  # noqa: E402
from paypal_client import PayPalClient, PayPalError, PayPalOrder  # noqa: E402
from policy import PolicyConfig  # noqa: E402
from schemas import LLMLineItem, LLMProcurementPlan  # noqa: E402
from service import ProcurementService  # noqa: E402


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def plan(*items: tuple[str, int], budget: float | None = None,
         unmatched: list[str] | None = None) -> LLMProcurementPlan:
    return LLMProcurementPlan(
        line_items=[LLMLineItem(sku=sku, quantity=qty, reason="test") for sku, qty in items],
        stated_budget=budget, unmatched_requests=unmatched or [], reasoning="test plan",
    )


def default_policy(**overrides) -> PolicyConfig:
    values = dict(
        auto_approve_limit=Decimal("150.00"), order_hard_cap=Decimal("350.00"),
        max_unit_price=Decimal("150.00"), monthly_budget=Decimal("1000.00"),
        max_quantity_per_line=50, approved_vendors=frozenset({"Acme Office Supply", "Globex Industrial"}),
    )
    values.update(overrides)
    return PolicyConfig(**values)


class ScriptedParser:
    """Returns pre-written plans in order (repeating the last); records the feedback it received."""

    def __init__(self, *responses: LLMProcurementPlan | Exception):
        self.responses = list(responses)
        self.calls: list[dict] = []
        self.last_model_used = "scripted"

    async def parse(self, request: str, catalog: Catalog, feedback: str | None = None) -> LLMProcurementPlan:
        self.calls.append({"request": request, "feedback": feedback})
        idx = min(len(self.calls), len(self.responses)) - 1
        response = self.responses[idx]
        if isinstance(response, Exception):
            raise response
        return response


class FakePayPal(PayPalClient):
    """In-memory PayPal. Vaulted orders complete on creation; redirect orders need `approve()`."""

    def __init__(self, *, fail_create: PayPalError | None = None, amount_override: str | None = None,
                 capture_amount_override: str | None = None, fail_capture: PayPalError | None = None):
        super().__init__(client_id="id", client_secret="secret", base_url="https://paypal.invalid",
                         http=None)  # type: ignore[arg-type]  # never used: network methods are overridden
        self.fail_create = fail_create
        self.amount_override = amount_override  # wrong amount echoed on create (vault: also captured)
        self.capture_amount_override = capture_amount_override  # wrong amount actually captured
        self.fail_capture = fail_capture
        self.orders: dict[str, dict] = {}
        self.capture_calls: list[str] = []
        self._approved: set[str] = set()

    async def create_vault_setup_token(self, *, return_url: str, cancel_url: str,
                                       request_id: str) -> tuple[str, str]:
        return "SETUP-1", "https://sandbox.paypal.invalid/agreements/approve?token=SETUP-1"

    def approve(self, order_id: str) -> None:
        self._approved.add(order_id)

    async def create_order(self, *, lines, total, currency, reference_id, description, payment_source,
                           request_id) -> PayPalOrder:
        if self.fail_create:
            raise self.fail_create
        oid = f"ORDER-{len(self.orders) + 1}"
        amount = self.amount_override or total
        self.orders[oid] = {"total": amount, "source": payment_source, "lines": list(lines),
                            "reference_id": reference_id, "request_id": request_id}
        if "vault_id" in payment_source["paypal"]:
            return PayPalOrder(id=oid, status="COMPLETED", amount=amount, currency=currency,
                               capture_id=f"CAP-{oid}", capture_status="COMPLETED")
        return PayPalOrder(id=oid, status="PAYER_ACTION_REQUIRED", amount=amount, currency=currency,
                           approval_url=f"https://sandbox.paypal.invalid/checkoutnow?token={oid}")

    async def capture_order(self, order_id: str, *, request_id: str) -> PayPalOrder:
        self.capture_calls.append(order_id)
        if order_id not in self._approved:
            raise PayPalError("Order not approved", status_code=422, name="UNPROCESSABLE_ENTITY",
                              details=[{"issue": "ORDER_NOT_APPROVED"}])
        if self.fail_capture:
            raise self.fail_capture
        amount = self.capture_amount_override or self.orders[order_id]["total"]
        return PayPalOrder(id=order_id, status="COMPLETED", amount=amount,
                           currency="USD", capture_id=f"CAP-{order_id}", capture_status="COMPLETED")


@pytest.fixture
def make_service():
    def _make(*responses, paypal: FakePayPal | None = None, vault_id: str | None = "VAULT-123",
              max_replans: int = 1, monthly_budget: str = "1000.00") -> ProcurementService:
        policy = default_policy(monthly_budget=Decimal(monthly_budget))
        return ProcurementService(
            parser=ScriptedParser(*responses), catalog=DEFAULT_CATALOG, policy=policy,
            ledger=SpendLedger(policy.monthly_budget), paypal=paypal or FakePayPal(),
            public_base_url="http://testserver", vault_id=vault_id, max_replans=max_replans,
        )
    return _make
