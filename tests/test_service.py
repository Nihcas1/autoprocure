"""Pipeline tests: plan -> policy -> (self-correct) -> PayPal, with a scripted planner and fake PayPal.

Prices (DEFAULT_CATALOG): TAPE-6PK 18.99, BC-SCAN-USB 64.99, BC-SCAN-WL 89.00,
BC-SCAN-2PK 159.00 (over unit cap), TH-ROLL-12 42.00, BC-SCAN-BUDGET 29.99 (unapproved vendor).
"""

from __future__ import annotations

import asyncio
from decimal import Decimal

import pytest

from agent import AgentError
from conftest import FakePayPal, plan
from paypal_client import PayPalError
from schemas import Decision, LLMLineItem, LLMProcurementPlan, ProcurementStatus

pytestmark = pytest.mark.anyio

SMALL = plan(("TAPE-6PK", 1), ("BC-SCAN-USB", 1))  # $83.98
MEDIUM = plan(("BC-SCAN-WL", 2), ("TH-ROLL-12", 2))  # $262.00
OVER_CAP_BUNDLE = plan(("BC-SCAN-2PK", 1), ("TH-ROLL-12", 2))  # blocked: unit_price_cap
UNAPPROVED = plan(("BC-SCAN-BUDGET", 1))  # blocked: vendor_approved
COMPLIANT_SCANNER = plan(("BC-SCAN-USB", 1))  # $64.99

PHONE_COMPLIANT = LLMProcurementPlan(
    line_items=[LLMLineItem(
        sku="MOB-EDGE-PRO", quantity=1, reason="matches 8gb ram and nfc",
        category="mobile-hardware", max_unit_price=150.0,
        required_specs={"ram_gb": 8, "has_nfc": True},
    )],
    stated_budget=150.0, unmatched_requests=[], reasoning="compliant phone",
)
PHONE_UNDER_SPEC = LLMProcurementPlan(
    line_items=[LLMLineItem(
        sku="MOB-BUDGET-01", quantity=1, reason="cheap phone",
        category="mobile-hardware", max_unit_price=150.0,
        required_specs={"ram_gb": 8, "has_nfc": True},
    )],
    stated_budget=150.0, unmatched_requests=[], reasoning="cheap phone failing specs",
)
PHONE_PRICE_VARIANCE = LLMProcurementPlan(
    line_items=[LLMLineItem(
        sku="MOB-EDGE-PRO", quantity=1, reason="matches specs but over target",
        category="mobile-hardware", max_unit_price=130.0,
        required_specs={"ram_gb": 8, "has_nfc": True},
    )],
    stated_budget=130.0, unmatched_requests=[], reasoning="compliant phone exceeding $130 target",
)
SCANNER_2D_FAIL = LLMProcurementPlan(
    line_items=[LLMLineItem(
        sku="BC-SCAN-BUDGET", quantity=1, reason="cheap scanner",
        category="pos-hardware", required_specs={"supports_2d": True},
    )],
    stated_budget=None, unmatched_requests=[], reasoning="1d scanner when 2d requested",
)


# ------------------------------------------------------------ auto-approval

async def test_auto_approved_order_is_captured_via_vault(make_service):
    paypal = FakePayPal()
    svc = make_service(SMALL, paypal=paypal)
    rec = await svc.procure("tape and a scanner")

    assert rec.policy.decision is Decision.AUTO_APPROVED
    assert rec.status is ProcurementStatus.PAID
    assert rec.payment.mode == "vault_auto_capture" and rec.payment.capture_id
    order = paypal.orders[rec.payment.paypal_order_id]
    assert order["source"] == {"paypal": {"vault_id": "VAULT-123"}}
    assert order["total"] == "83.98"
    assert svc.ledger.snapshot()["committed"] == "83.98"
    assert len(rec.attempts) == 1


async def test_auto_approved_without_mandate_falls_back_to_human_approval(make_service):
    svc = make_service(SMALL, vault_id=None)
    rec = await svc.procure("tape and a scanner")

    assert rec.policy.decision is Decision.AUTO_APPROVED
    assert rec.status is ProcurementStatus.AWAITING_APPROVAL
    assert rec.payment.mode == "manager_approval" and rec.payment.approval_url
    assert "PAYPAL_VAULT_ID" in rec.payment.note


# ------------------------------------------------------ manager approval

async def test_manager_approval_flow(make_service):
    paypal = FakePayPal()
    svc = make_service(MEDIUM, paypal=paypal)
    rec = await svc.procure("2 wireless scanners and 24 rolls")
    oid = rec.payment.paypal_order_id

    assert rec.policy.decision is Decision.NEEDS_APPROVAL
    assert rec.status is ProcurementStatus.AWAITING_APPROVAL
    assert svc.ledger.snapshot()["reserved"] == "262.00"

    # Return link hit before the manager approved: stays pending, budget stays reserved.
    rec = await svc.complete_approval(oid)
    assert rec.status is ProcurementStatus.AWAITING_APPROVAL
    assert svc.ledger.snapshot()["reserved"] == "262.00"

    paypal.approve(oid)
    rec = await svc.complete_approval(oid)
    assert rec.status is ProcurementStatus.PAID
    assert svc.ledger.snapshot()["committed"] == "262.00"

    # Idempotent: a second redirect does not capture again.
    calls = len(paypal.capture_calls)
    await svc.complete_approval(oid)
    assert len(paypal.capture_calls) == calls


async def test_cancel_releases_budget(make_service):
    svc = make_service(MEDIUM)
    rec = await svc.procure("2 wireless scanners and 24 rolls")
    rec = await svc.cancel_approval(rec.payment.paypal_order_id)
    assert rec.status is ProcurementStatus.CANCELLED
    assert svc.ledger.remaining() == Decimal("1000.00")


# ------------------------------------------------- rejection -> recovery

async def test_rejected_plan_self_corrects_to_manager_approval(make_service):
    svc = make_service(OVER_CAP_BUNDLE, MEDIUM)
    rec = await svc.procure("2 wireless scanners and 24 rolls under $300")

    assert [a.policy.decision for a in rec.attempts] == [Decision.BLOCKED, Decision.NEEDS_APPROVAL]
    assert rec.attempts[0].feedback is None
    assert "[unit_price_cap]" in rec.attempts[1].feedback
    assert "1 x BC-SCAN-2PK" in svc.parser.calls[1]["feedback"]
    assert rec.status is ProcurementStatus.AWAITING_APPROVAL
    assert rec.policy.subtotal == Decimal("262.00")
    assert svc.ledger.snapshot()["reserved"] == "262.00"  # only the final plan reserves budget


async def test_rejected_plan_self_corrects_to_auto_payment(make_service):
    svc = make_service(UNAPPROVED, COMPLIANT_SCANNER)
    rec = await svc.procure("the cheapest barcode scanner")

    assert len(rec.attempts) == 2
    assert "[vendor_approved]" in rec.attempts[1].feedback
    assert rec.status is ProcurementStatus.PAID
    assert rec.policy.subtotal == Decimal("64.99")


async def test_recovery_that_fails_again_ends_blocked(make_service):
    paypal = FakePayPal()
    svc = make_service(OVER_CAP_BUNDLE, OVER_CAP_BUNDLE, paypal=paypal)
    rec = await svc.procure("the scanner bundle")

    assert len(rec.attempts) == 2
    assert rec.status is ProcurementStatus.BLOCKED
    assert rec.exemption_request and "unit_price_cap" in rec.exemption_request
    assert not paypal.orders
    assert svc.ledger.remaining() == Decimal("1000.00")


@pytest.mark.parametrize("max_replans,expected_attempts", [(0, 1), (1, 2), (2, 3)])
async def test_replan_budget_is_respected(make_service, max_replans, expected_attempts):
    svc = make_service(OVER_CAP_BUNDLE, max_replans=max_replans)
    rec = await svc.procure("the scanner bundle")
    assert len(rec.attempts) == expected_attempts
    assert len(svc.parser.calls) == expected_attempts


async def test_uncorrectable_block_does_not_replan(make_service):
    svc = make_service(plan(unmatched=["3 coffee machines"]))
    rec = await svc.procure("3 coffee machines")
    assert rec.status is ProcurementStatus.BLOCKED
    assert len(rec.attempts) == 1


async def test_replan_agent_failure_keeps_the_blocked_result(make_service):
    svc = make_service(OVER_CAP_BUNDLE, AgentError("model overloaded"))
    rec = await svc.procure("the scanner bundle")
    assert rec.status is ProcurementStatus.BLOCKED
    assert len(rec.attempts) == 1


async def test_first_plan_failure_propagates(make_service):
    svc = make_service(AgentError("model overloaded"))
    with pytest.raises(AgentError):
        await svc.procure("anything")
    assert not svc.records


# --------------------------------------------------- payment-side failures

async def test_paypal_amount_mismatch_fails_safe(make_service):
    svc = make_service(MEDIUM, paypal=FakePayPal(amount_override="999.00"))
    rec = await svc.procure("2 wireless scanners and 24 rolls")
    assert rec.status is ProcurementStatus.PAYMENT_FAILED
    assert "does not match" in rec.error
    assert svc.ledger.remaining() == Decimal("1000.00")


async def test_vault_capture_at_wrong_amount_is_flagged_and_budget_kept(make_service):
    # Vaulted orders are captured on creation: a mismatch here means money already moved.
    svc = make_service(SMALL, paypal=FakePayPal(amount_override="95.00"))
    rec = await svc.procure("tape and a scanner")
    assert rec.status is ProcurementStatus.NEEDS_REVIEW
    assert "does not match" in rec.error
    assert svc.ledger.snapshot()["committed"] == "95.00"  # what PayPal actually took
    assert svc.ledger.remaining() == Decimal("905.00")


async def test_approved_capture_at_wrong_amount_is_flagged_and_budget_kept(make_service):
    paypal = FakePayPal(capture_amount_override="300.00")
    svc = make_service(MEDIUM, paypal=paypal)
    rec = await svc.procure("2 wireless scanners and 24 rolls")
    paypal.approve(rec.payment.paypal_order_id)
    rec = await svc.complete_approval(rec.payment.paypal_order_id)
    assert rec.status is ProcurementStatus.NEEDS_REVIEW
    assert rec.payment.capture_id  # the capture is recorded for finance
    assert svc.ledger.snapshot()["committed"] == "300.00"
    assert svc.ledger.remaining() == Decimal("700.00")


@pytest.mark.parametrize("status_code", [None, 500, 503])
async def test_capture_with_unknown_outcome_keeps_budget_reserved(make_service, status_code):
    paypal = FakePayPal(fail_capture=PayPalError("timeout", status_code=status_code))
    svc = make_service(MEDIUM, paypal=paypal)
    rec = await svc.procure("2 wireless scanners and 24 rolls")
    paypal.approve(rec.payment.paypal_order_id)
    rec = await svc.complete_approval(rec.payment.paypal_order_id)
    assert rec.status is ProcurementStatus.NEEDS_REVIEW
    assert svc.ledger.snapshot()["reserved"] == "262.00"  # never freed while money may have moved


async def test_capture_definitively_declined_releases_budget(make_service):
    paypal = FakePayPal(fail_capture=PayPalError("INSTRUMENT_DECLINED", status_code=422))
    svc = make_service(MEDIUM, paypal=paypal)
    rec = await svc.procure("2 wireless scanners and 24 rolls")
    paypal.approve(rec.payment.paypal_order_id)
    rec = await svc.complete_approval(rec.payment.paypal_order_id)
    assert rec.status is ProcurementStatus.PAYMENT_FAILED
    assert svc.ledger.remaining() == Decimal("1000.00")


async def test_vault_payment_with_unknown_outcome_is_flagged(make_service):
    svc = make_service(SMALL, paypal=FakePayPal(fail_create=PayPalError("read timeout", status_code=None)))
    rec = await svc.procure("tape and a scanner")
    assert rec.status is ProcurementStatus.NEEDS_REVIEW
    assert svc.ledger.snapshot()["reserved"] == "83.98"


async def test_definitive_vault_decline_releases_reservation(make_service):
    svc = make_service(SMALL, paypal=FakePayPal(fail_create=PayPalError("declined", status_code=422)))
    rec = await svc.procure("tape and a scanner")
    assert rec.status is ProcurementStatus.PAYMENT_FAILED
    assert svc.ledger.remaining() == Decimal("1000.00")


async def test_redirect_order_failure_releases_even_on_5xx(make_service):
    # Creating an approval-link order can never move money, so any failure is safe to release.
    svc = make_service(MEDIUM, paypal=FakePayPal(fail_create=PayPalError("boom", status_code=500)))
    rec = await svc.procure("2 wireless scanners and 24 rolls")
    assert rec.status is ProcurementStatus.PAYMENT_FAILED
    assert svc.ledger.remaining() == Decimal("1000.00")


async def test_concurrent_requests_cannot_overspend_monthly_budget(make_service):
    svc = make_service(MEDIUM, monthly_budget="300.00", max_replans=0)
    a, b = await asyncio.gather(svc.procure("order A"), svc.procure("order B"))
    statuses = sorted([a.status, b.status])
    assert statuses == [ProcurementStatus.AWAITING_APPROVAL, ProcurementStatus.BLOCKED]
    blocked = a if a.status is ProcurementStatus.BLOCKED else b
    assert {c.rule for c in blocked.policy.failures} >= {"monthly_budget"}


# ---------------------------------------------------- technical specifications

async def test_spec_compliant_phone_auto_approved_and_captured(make_service):
    paypal = FakePayPal()
    svc = make_service(PHONE_COMPLIANT, paypal=paypal)
    rec = await svc.procure("Order 1 warehouse phone with at least 8GB RAM and NFC under $150")

    assert rec.policy.decision is Decision.AUTO_APPROVED
    assert rec.status is ProcurementStatus.PAID
    assert rec.policy.subtotal == Decimal("145.00")
    assert rec.payment.mode == "vault_auto_capture"


async def test_under_spec_phone_self_corrects_to_compliant_phone(make_service):
    paypal = FakePayPal()
    svc = make_service(PHONE_UNDER_SPEC, PHONE_COMPLIANT, paypal=paypal)
    rec = await svc.procure("Order 1 warehouse phone with at least 8GB RAM and NFC under $150")

    assert len(rec.attempts) == 2
    assert rec.attempts[0].policy.decision is Decision.BLOCKED
    assert "spec_compliance" in {c.rule for c in rec.attempts[0].policy.failures}
    assert "[spec_compliance]" in rec.attempts[1].feedback
    assert rec.status is ProcurementStatus.PAID
    assert rec.policy.decision is Decision.AUTO_APPROVED
    assert rec.policy.subtotal == Decimal("145.00")


async def test_spec_compliant_phone_over_target_price_routes_to_manager_approval(make_service):
    paypal = FakePayPal()
    svc = make_service(PHONE_PRICE_VARIANCE, paypal=paypal)
    rec = await svc.procure("Order 1 warehouse phone with 8GB RAM and NFC under $130")

    assert rec.policy.decision is Decision.NEEDS_APPROVAL
    assert rec.status is ProcurementStatus.AWAITING_APPROVAL
    assert "unit_price_variance" in {c.rule for c in rec.policy.failures}
    assert rec.policy.subtotal == Decimal("145.00")
    assert rec.payment.mode == "manager_approval"
    assert rec.payment.approval_url


async def test_scanner_2d_requirement_blocks_1d_scanner(make_service):
    svc = make_service(SCANNER_2D_FAIL, max_replans=0)
    rec = await svc.procure("Order 1 corded USB scanner that reads QR codes")

    assert rec.policy.decision is Decision.BLOCKED
    assert rec.status is ProcurementStatus.BLOCKED
    assert "spec_compliance" in {c.rule for c in rec.policy.failures}
    assert any("supports_2d=False" in c.detail for c in rec.policy.failures if c.rule == "spec_compliance")
