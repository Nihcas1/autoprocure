"""Deterministic guardrail engine: decisions, boundaries, and exact money math."""

from __future__ import annotations

from decimal import Decimal

import pytest

from catalog import Catalog, CatalogItem, DEFAULT_CATALOG
from conftest import default_policy, plan
from policy import extract_budget, evaluate, is_correctable, replan_feedback, validate_product_specs
from schemas import Decision, LLMLineItem, LLMProcurementPlan

ACME, SHADY = "Acme Office Supply", "Shady Deals"

CATALOG = Catalog([
    CatalogItem(sku="A50", name="Item at $50", vendor=ACME, category="t", unit_price=Decimal("50.00"), description=""),
    CatalogItem(sku="A150", name="Item at $150", vendor=ACME, category="t", unit_price=Decimal("150.00"), description=""),
    CatalogItem(sku="A15001", name="Item at $150.01", vendor=ACME, category="t", unit_price=Decimal("150.01"), description=""),
    CatalogItem(sku="PENNY", name="Penny item", vendor=ACME, category="t", unit_price=Decimal("0.01"), description=""),
    CatalogItem(sku="DIME", name="Ten cents", vendor=ACME, category="t", unit_price=Decimal("0.10"), description=""),
    CatalogItem(sku="SHADY10", name="Unapproved vendor item", vendor=SHADY, category="t", unit_price=Decimal("10.00"), description=""),
])


def run(p, request_text: str = "test request", remaining: str = "1000.00", **policy_overrides):
    return evaluate(p, request_text=request_text, catalog=CATALOG, config=default_policy(**policy_overrides),
                    monthly_remaining=Decimal(remaining))


def failed_rules(result) -> set[str]:
    return {c.rule for c in result.failures}


# ----------------------------------------------------------------- decisions

def test_small_compliant_order_is_auto_approved():
    r = run(plan(("A50", 2)))
    assert r.decision is Decision.AUTO_APPROVED
    assert r.subtotal == Decimal("100.00")
    assert not r.failures


def test_exactly_auto_limit_needs_approval():
    r = run(plan(("A50", 3)))  # $150.00 is NOT below the $150 auto limit
    assert r.decision is Decision.NEEDS_APPROVAL
    assert failed_rules(r) == {"auto_approval_threshold"}


def test_one_cent_below_auto_limit_is_auto_approved():
    r = run(plan(("A50", 2), ("DIME", 499), ("PENNY", 9)), max_quantity_per_line=500)  # 100 + 49.90 + 0.09
    assert r.subtotal == Decimal("149.99")
    assert r.decision is Decision.AUTO_APPROVED


def test_exactly_hard_cap_is_allowed_with_approval():
    r = run(plan(("A50", 7)))
    assert r.subtotal == Decimal("350.00")
    assert r.decision is Decision.NEEDS_APPROVAL


def test_one_cent_over_hard_cap_is_blocked():
    r = run(plan(("A50", 7), ("PENNY", 1)))
    assert r.subtotal == Decimal("350.01")
    assert r.decision is Decision.BLOCKED
    assert "order_hard_cap" in failed_rules(r)


# ------------------------------------------------------------- hard blocks

def test_unit_price_cap_boundary():
    assert "unit_price_cap" not in failed_rules(run(plan(("A150", 1))))
    r = run(plan(("A15001", 1)))
    assert r.decision is Decision.BLOCKED and "unit_price_cap" in failed_rules(r)


def test_unapproved_vendor_is_blocked():
    r = run(plan(("SHADY10", 1)))
    assert r.decision is Decision.BLOCKED
    assert failed_rules(r) == {"vendor_approved"}


def test_hallucinated_sku_is_blocked_and_never_priced():
    r = run(plan(("A50", 1), ("MAGIC-9000", 1)))
    assert r.decision is Decision.BLOCKED
    assert "sku_in_catalog" in failed_rules(r)
    assert [l.sku for l in r.lines] == ["A50"]


@pytest.mark.parametrize("qty", [0, -3, 51])
def test_quantity_out_of_range_is_blocked(qty):
    r = run(plan(("A50", qty)))
    assert r.decision is Decision.BLOCKED
    assert "quantity_within_limits" in failed_rules(r)


def test_negative_quantity_cannot_reduce_the_total():
    r = run(plan(("A50", 3), ("PENNY", -10000)))
    assert r.subtotal == Decimal("150.00")
    assert r.decision is Decision.BLOCKED


def test_negative_duplicate_line_cannot_net_against_positive():
    r = run(plan(("A50", 5), ("A50", -3)))
    assert r.decision is Decision.BLOCKED
    assert "quantity_within_limits" in failed_rules(r)


def test_empty_plan_is_blocked():
    r = run(plan())
    assert r.decision is Decision.BLOCKED
    assert "has_line_items" in failed_rules(r)


def test_monthly_budget_is_enforced():
    r = run(plan(("A50", 2)), remaining="99.99")
    assert r.decision is Decision.BLOCKED
    assert failed_rules(r) == {"monthly_budget"}


def test_every_rule_is_reported_not_just_the_first():
    r = run(plan(("SHADY10", 1), ("A15001", 3), ("NOPE", 1)))
    assert {"vendor_approved", "unit_price_cap", "sku_in_catalog", "order_hard_cap"} <= failed_rules(r)


# ------------------------------------------------------- review (human) flags

def test_unmatched_request_routes_small_order_to_a_human():
    r = run(plan(("A50", 1), unmatched=["a coffee machine"]))
    assert r.decision is Decision.NEEDS_APPROVAL
    assert "all_requests_matched" in failed_rules(r)


# ------------------------------------------------------------------- money

def test_money_math_is_exact():
    r = run(plan(("DIME", 3)))  # 0.1 * 3 is 0.30000000000000004 in float
    assert r.subtotal == Decimal("0.30")


def test_duplicate_skus_are_merged():
    r = run(plan(("A50", 1), ("a50 ", 2)))
    assert len(r.lines) == 1 and r.lines[0].quantity == 3
    assert r.subtotal == Decimal("150.00")


# --------------------------------------------------------- requester budget

@pytest.mark.parametrize("text,expected", [
    ("Restock 5 thermal rolls & 2 scanners under $300", "300.00"),
    ("budget of $1,200.50 for boxes", "1200.50"),
    ("max $80 please", "80.00"),
    ("$45 budget, need tape", "45.00"),
    ("no more than $90 and below $75", "75.00"),  # strictest wins
    ("buy 2 scanners at $5 each", None),  # a unit price is not a budget
    ("buy 2 scanners", None),
])
def test_extract_budget(text, expected):
    assert extract_budget(text) == (Decimal(expected) if expected else None)


def test_request_text_budget_overrides_a_looser_llm_reading():
    r = run(plan(("A50", 2), budget=500.0), request_text="two items under $90")
    assert r.effective_budget == Decimal("90.00")
    assert r.decision is Decision.BLOCKED
    assert "within_requester_budget" in failed_rules(r)


def test_llm_budget_used_when_text_has_no_dollar_figure():
    r = run(plan(("A50", 2), budget=90.0), request_text="two items, keep it under ninety dollars")
    assert r.effective_budget == Decimal("90.00")
    assert r.decision is Decision.BLOCKED


# --------------------------------------------------------- self-correction

def test_correctable_vs_not():
    assert is_correctable(run(plan(("A15001", 1))))  # pick a cheaper SKU
    assert is_correctable(run(plan(("SHADY10", 1))))  # pick an approved vendor
    assert not is_correctable(run(plan()))  # nothing matched: retrying won't help
    assert not is_correctable(run(plan(("A50", 1))))  # not blocked at all


def test_replan_feedback_lists_violations_and_rejected_items():
    p = plan(("A15001", 1), ("SHADY10", 2))
    fb = replan_feedback(p, run(p))
    assert "[unit_price_cap]" in fb and "[vendor_approved]" in fb
    assert "1 x A15001" in fb and "2 x SHADY10" in fb
    assert "unmatched_requests" in fb


# ---------------------------------------------------- technical specifications

def test_validate_product_specs_numeric_thresholds():
    phone = DEFAULT_CATALOG.get("MOB-EDGE-PRO")
    assert phone is not None
    # ram_gb: 8, battery_mah: 5000
    ok, issues = validate_product_specs(phone, {"ram_gb": 8, "battery_mah": 4500})
    assert ok and not issues

    ok, issues = validate_product_specs(phone, {"ram_gb": 12})
    assert not ok
    assert any("ram_gb=8 < required 12" in i for i in issues)


def test_validate_product_specs_boolean_and_lists():
    scanner = DEFAULT_CATALOG.get("BC-SCAN-USB")
    assert scanner is not None
    # supports_2d: True, connectivity: usb
    ok, issues = validate_product_specs(scanner, {"supports_2d": True, "connectivity": "usb"})
    assert ok and not issues

    budget_scanner = DEFAULT_CATALOG.get("BC-SCAN-BUDGET")
    assert budget_scanner is not None
    ok, issues = validate_product_specs(budget_scanner, {"supports_2d": True})
    assert not ok
    assert any("supports_2d=False" in i for i in issues)


def test_validate_product_specs_missing_spec_fails():
    tape = DEFAULT_CATALOG.get("TAPE-6PK")
    assert tape is not None
    ok, issues = validate_product_specs(tape, {"has_nfc": True})
    assert not ok
    assert any("missing spec 'has_nfc'" in i for i in issues)


def test_spec_compliance_matching_specs_passes():
    line = LLMLineItem(
        sku="MOB-EDGE-PRO", quantity=1, reason="matches 8gb ram and nfc",
        required_specs={"ram_gb": 8, "has_nfc": True}, max_unit_price=150.0,
    )
    p = LLMProcurementPlan(line_items=[line], stated_budget=150.0, unmatched_requests=[], reasoning="ok")
    r = evaluate(p, request_text="warehouse phone with 8GB RAM and NFC under $150",
                 catalog=DEFAULT_CATALOG, config=default_policy(), monthly_remaining=Decimal("1000.00"))
    assert r.decision is Decision.AUTO_APPROVED
    assert r.subtotal == Decimal("145.00")
    assert not r.failures


def test_spec_compliance_mismatch_blocks_order():
    line = LLMLineItem(
        sku="MOB-BUDGET-01", quantity=1, reason="cheap phone",
        required_specs={"ram_gb": 8, "has_nfc": True}, max_unit_price=130.0,
    )
    p = LLMProcurementPlan(line_items=[line], stated_budget=130.0, unmatched_requests=[], reasoning="wrong phone")
    r = evaluate(p, request_text="warehouse phone with 8GB RAM and NFC under $130",
                 catalog=DEFAULT_CATALOG, config=default_policy(), monthly_remaining=Decimal("1000.00"))
    assert r.decision is Decision.BLOCKED
    assert "spec_compliance" in failed_rules(r)
    assert is_correctable(r)


def test_spec_matches_but_unit_price_exceeds_target_routes_to_manager_review():
    line = LLMLineItem(
        sku="MOB-EDGE-PRO", quantity=1, reason="compliant phone",
        required_specs={"ram_gb": 8, "has_nfc": True}, max_unit_price=130.0,
    )
    p = LLMProcurementPlan(line_items=[line], stated_budget=130.0, unmatched_requests=[], reasoning="price variance")
    r = evaluate(p, request_text="warehouse phone with 8GB RAM and NFC under $130",
                 catalog=DEFAULT_CATALOG, config=default_policy(), monthly_remaining=Decimal("1000.00"))
    assert r.decision is Decision.NEEDS_APPROVAL
    assert "unit_price_variance" in failed_rules(r)
    assert "within_requester_budget" in failed_rules(r)
    # Spec compliance itself passed
    assert "spec_compliance" not in failed_rules(r)
