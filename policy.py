"""Deterministic guardrail engine.

Pure functions over `Decimal`. The LLM's plan is treated as untrusted input:
prices come from the catalog, totals are computed here, and every rule is
evaluated (not short-circuited) so the UI can show a full checklist.
"""

from __future__ import annotations

import re
from collections import OrderedDict
from decimal import Decimal
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from catalog import Catalog, CatalogItem
from money import to_money
from schemas import (
    Decision,
    LLMProcurementPlan,
    PolicyCheck,
    PolicyResult,
    PricedLine,
    Severity,
)


class PolicyConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    auto_approve_limit: Decimal = Field(gt=0)
    order_hard_cap: Decimal = Field(gt=0)
    max_unit_price: Decimal = Field(gt=0)
    monthly_budget: Decimal = Field(gt=0)
    max_quantity_per_line: int = Field(ge=1)
    approved_vendors: frozenset[str]

    @model_validator(mode="after")
    def _consistent(self) -> "PolicyConfig":
        if self.auto_approve_limit > self.order_hard_cap:
            raise ValueError("auto_approve_limit must not exceed order_hard_cap")
        return self


# "under $300", "budget of $1,200.50", "max $80", "no more than $45", "<= $90", "$300 budget", "$300 max"
_AMOUNT = r"\$\s?(\d{1,3}(?:,\d{3})+(?:\.\d{1,2})?|\d+(?:\.\d{1,2})?)"
_BUDGET_PATTERNS = [
    re.compile(
        r"(?:under|below|less than|at most|max(?:imum)?(?: of)?|budget(?: of| is)?|up to|within|"
        r"no more than|not (?:to )?exceed(?:ing)?|cap(?: of)?|<=?)\s*(?:a\s+)?(?:US)?" + _AMOUNT,
        re.IGNORECASE,
    ),
    re.compile(_AMOUNT + r"\s*(?:budget|max(?:imum)?|cap|limit|or less)\b", re.IGNORECASE),
]


def extract_budget(text: str) -> Decimal | None:
    """Deterministically pull the strictest stated budget out of the request text."""
    found = [
        to_money(m.group(1).replace(",", ""))
        for pattern in _BUDGET_PATTERNS
        for m in pattern.finditer(text)
    ]
    found = [a for a in found if a > 0]
    return min(found) if found else None


def _money_str(d: Decimal) -> str:
    return f"${to_money(d):,.2f}"


def validate_product_specs(product: CatalogItem, requirement: dict[str, Any] | Any) -> tuple[bool, list[str]]:
    """Validate a catalog product against required technical specifications.

    - Numeric thresholds: int/float verified with >= required_val.
    - Boolean / Exact string matches: exact equality (or list membership if product spec is a list).
    - Missing keys in product: treated as a mismatch.

    Returns (passed, list_of_mismatch_details).
    """
    required_specs = getattr(requirement, "required_specs", requirement)
    mismatches: list[str] = []
    for key, req_val in (required_specs or {}).items():
        if key not in product.specs:
            mismatches.append(f"missing spec '{key}' (required: {req_val!r})")
            continue
        prod_val = product.specs[key]

        # Numeric threshold (int/float, excluding bool)
        if isinstance(req_val, (int, float)) and not isinstance(req_val, bool):
            if not isinstance(prod_val, (int, float)) or isinstance(prod_val, bool) or prod_val < req_val:
                mismatches.append(f"{key}={prod_val} < required {req_val}")
        # Boolean match
        elif isinstance(req_val, bool):
            if prod_val is not req_val:
                mismatches.append(f"{key}={prod_val} (required: {req_val})")
        # List membership (e.g. connectivity: ["usb", "ethernet"] matches "usb")
        elif isinstance(prod_val, list):
            req_str = str(req_val).lower()
            if req_val not in prod_val and req_str not in [str(x).lower() for x in prod_val]:
                mismatches.append(f"{key}={prod_val} does not include {req_val!r}")
        # String match (case-insensitive)
        elif isinstance(req_val, str):
            if str(prod_val).lower() != req_val.lower():
                mismatches.append(f"{key}={prod_val!r} (required: {req_val!r})")
        else:
            if prod_val != req_val:
                mismatches.append(f"{key}={prod_val!r} (required: {req_val!r})")
    return not mismatches, mismatches


def evaluate(
    plan: LLMProcurementPlan,
    *,
    request_text: str,
    catalog: Catalog,
    config: PolicyConfig,
    monthly_remaining: Decimal,
    currency: str = "USD",
) -> PolicyResult:
    checks: list[PolicyCheck] = []

    def check(rule: str, passed: bool, severity: Severity, detail: str) -> None:
        checks.append(PolicyCheck(rule=rule, passed=passed, severity=severity, detail=detail))

    # 1. Merge duplicate SKUs (the LLM may list the same SKU twice).
    merged: OrderedDict[str, int] = OrderedDict()
    for li in plan.line_items:
        key = Catalog.normalize(li.sku)
        merged[key] = merged.get(key, 0) + li.quantity

    check("has_line_items", bool(merged), Severity.BLOCK,
          f"{len(merged)} distinct SKU(s) proposed." if merged else "The agent proposed no purchasable items.")

    # 2. Every SKU must exist in the catalog (catches hallucinated SKUs).
    unknown = [sku for sku in merged if sku not in catalog]
    check("sku_in_catalog", not unknown, Severity.BLOCK,
          "All SKUs exist in the catalog." if not unknown
          else f"Unknown SKU(s), possibly hallucinated: {', '.join(unknown)}.")

    lines: list[PricedLine] = []
    for sku, qty in merged.items():
        item = catalog.get(sku)
        if item is None:
            continue
        unit = to_money(item.unit_price)
        lines.append(PricedLine(sku=item.sku, name=item.name, vendor=item.vendor,
                                unit_price=unit, quantity=qty, line_total=to_money(unit * qty)))

    # 3. Quantities. Each RAW line is checked before merging too, so a negative line can't
    #    net against a positive duplicate (5 + -3 = 2 must not pass).
    bad_qty = [f"{li.sku} x{li.quantity}" for li in plan.line_items if li.quantity < 1]
    bad_qty += [f"{l.sku} x{l.quantity}" for l in lines if l.quantity > config.max_quantity_per_line]
    check("quantity_within_limits", not bad_qty, Severity.BLOCK,
          f"All quantities are between 1 and {config.max_quantity_per_line}." if not bad_qty
          else f"Quantity out of range (1-{config.max_quantity_per_line}): {', '.join(bad_qty)}.")

    # 4. Vendor whitelist.
    bad_vendor = sorted({l.vendor for l in lines if l.vendor not in config.approved_vendors})
    check("vendor_approved", not bad_vendor, Severity.BLOCK,
          "All vendors are on the approved list." if not bad_vendor
          else f"Vendor(s) not on the approved list: {', '.join(bad_vendor)}.")

    # 5. Per-unit price cap.
    pricey = [f"{l.sku} at {_money_str(l.unit_price)}" for l in lines if l.unit_price > config.max_unit_price]
    check("unit_price_cap", not pricey, Severity.BLOCK,
          f"Every unit costs at most {_money_str(config.max_unit_price)}." if not pricey
          else f"Unit price above the {_money_str(config.max_unit_price)} cap: {', '.join(pricey)}.")

    # Only count lines with valid quantities toward the subtotal; invalid ones already block.
    subtotal = to_money(sum((l.line_total for l in lines if l.quantity >= 1), Decimal("0")))

    # 6. Technical specification validation (Zero-Guess Rule).
    spec_failures: list[str] = []
    price_variances: list[str] = []
    for li in plan.line_items:
        item = catalog.get(li.sku)
        if item is None:
            continue
        reqs = getattr(li, "required_specs", None) or {}
        if reqs:
            ok, issues = validate_product_specs(item, reqs)
            if not ok:
                spec_failures.append(f"{item.sku}: {'; '.join(issues)}")
        target_price = getattr(li, "max_unit_price", None)
        if target_price is not None and target_price > 0:
            target = to_money(target_price)
            actual = to_money(item.unit_price)
            if actual > target:
                price_variances.append(
                    f"{item.sku} ({_money_str(actual)}) exceeds target {_money_str(target)}"
                )

    check("spec_compliance", not spec_failures, Severity.BLOCK,
          "All items satisfy required technical specifications." if not spec_failures
          else f"Technical specification mismatch: {'; '.join(spec_failures)}.")

    check("unit_price_variance", not price_variances, Severity.REVIEW,
          "All items are within requested unit price targets." if not price_variances
          else f"Target unit price variance: {'; '.join(price_variances)}.")

    # 7. Requester's own budget. The regex reading of the raw text and the LLM's
    #    reading are both honored; the stricter one wins (a budget can only restrict).
    candidates: list[Decimal] = []
    text_budget = extract_budget(request_text)
    if text_budget is not None:
        candidates.append(text_budget)
    if plan.stated_budget is not None and plan.stated_budget > 0:
        try:
            candidates.append(to_money(plan.stated_budget))
        except ValueError:
            pass
    effective_budget = min(candidates) if candidates else None
    if effective_budget is None:
        check("within_requester_budget", True, Severity.BLOCK, "No budget stated by the requester.")
    else:
        ok = subtotal <= effective_budget
        # When an item meets technical specifications but exceeds the user's unit/order budget
        # (while remaining within corporate caps), route to REVIEW as a price variance sign-off.
        budget_severity = Severity.REVIEW if (price_variances and not spec_failures) else Severity.BLOCK
        check("within_requester_budget", ok, budget_severity,
              f"Total {_money_str(subtotal)} {'is within' if ok else 'exceeds'} the requested budget "
              f"of {_money_str(effective_budget)}.")

    # 8. Per-order hard cap.
    ok = subtotal <= config.order_hard_cap
    check("order_hard_cap", ok, Severity.BLOCK,
          f"Total {_money_str(subtotal)} {'is within' if ok else 'exceeds'} the per-order cap of "
          f"{_money_str(config.order_hard_cap)}.")

    # 9. Department monthly budget.
    ok = subtotal <= monthly_remaining
    check("monthly_budget", ok, Severity.BLOCK,
          f"{_money_str(monthly_remaining)} left this month; this order "
          f"{'fits' if ok else 'does not fit'}.")

    # 10. Everything the requester asked for was matched (partial orders get a human look).
    check("all_requests_matched", not plan.unmatched_requests, Severity.REVIEW,
          "Every part of the request was matched to a catalog item." if not plan.unmatched_requests
          else f"Could not source: {'; '.join(plan.unmatched_requests)}.")

    # 11. Auto-approval threshold.
    ok = subtotal < config.auto_approve_limit
    check("auto_approval_threshold", ok, Severity.REVIEW,
          f"Total {_money_str(subtotal)} is {'below' if ok else 'at or above'} the auto-approval "
          f"limit of {_money_str(config.auto_approve_limit)}.")

    failed = [c for c in checks if not c.passed]
    if any(c.severity is Severity.BLOCK for c in failed):
        decision = Decision.BLOCKED
    elif failed:
        decision = Decision.NEEDS_APPROVAL
    else:
        decision = Decision.AUTO_APPROVED

    return PolicyResult(decision=decision, checks=checks, lines=lines, subtotal=subtotal,
                        currency=currency, effective_budget=effective_budget)


# Blocking rules the planner can plausibly fix by choosing different SKUs/quantities.
# `has_line_items` is deliberately excluded: if nothing in the catalog matches, retrying won't help.
CORRECTABLE_RULES = frozenset({
    "sku_in_catalog", "quantity_within_limits", "vendor_approved", "unit_price_cap",
    "within_requester_budget", "order_hard_cap", "monthly_budget", "spec_compliance",
})


def is_correctable(result: PolicyResult) -> bool:
    """True if the plan was blocked only by rules a re-plan could satisfy."""
    blockers = [c for c in result.failures if c.severity is Severity.BLOCK]
    return bool(blockers) and all(c.rule in CORRECTABLE_RULES for c in blockers)


def replan_feedback(plan: LLMProcurementPlan, result: PolicyResult) -> str:
    """Deterministic feedback handed back to the planner after a policy rejection."""
    blockers = [c for c in result.failures if c.severity is Severity.BLOCK]
    rejected = "\n".join(f"- {li.quantity} x {li.sku}" for li in plan.line_items) or "- (none)"
    reasons = "\n".join(f"- [{c.rule}] {c.detail}" for c in blockers)
    return (
        "Your previous plan was REJECTED by the deterministic policy engine.\n"
        f"Rejected plan:\n{rejected}\n"
        f"Violations:\n{reasons}\n"
        "Produce a corrected plan that satisfies every policy rule. Prefer compliant alternatives "
        "(e.g. individual units instead of an over-cap bundle, an approved vendor's equivalent). "
        "If part of the request cannot be fulfilled compliantly, leave it out and list it in "
        "`unmatched_requests`; never silently buy less than was asked."
    )


def draft_exemption_request(result: PolicyResult, request_text: str, request_id: str) -> str:
    """Deterministic, human-editable exemption memo for a blocked purchase."""
    blockers = [c for c in result.failures if c.severity is Severity.BLOCK]
    lines = "\n".join(f"  - {l.quantity} x {l.name} ({l.sku}, {l.vendor}) @ {_money_str(l.unit_price)}"
                      for l in result.lines) or "  - (no priced items)"
    reasons = "\n".join(f"  - [{c.rule}] {c.detail}" for c in blockers)
    return (
        f"Subject: Policy exemption request - {request_id}\n\n"
        f"Original request: \"{request_text}\"\n\n"
        f"Proposed items:\n{lines}\n"
        f"Proposed total: {_money_str(result.subtotal)} {result.currency}\n\n"
        f"Blocked by:\n{reasons}\n\n"
        f"Business justification: <requester to complete>\n"
        f"Approver: <finance / department head>\n"
    )
