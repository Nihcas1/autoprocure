"""Pydantic schemas shared across the agent, policy engine, PayPal layer and API."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# LLM output contract. The model chooses SKUs and quantities ONLY; it never
# produces prices or totals. Every field is required so the structured output
# is unambiguous.
# ---------------------------------------------------------------------------


class LLMLineItem(BaseModel):
    sku: str = Field(description="Exact SKU copied verbatim from the catalog.")
    quantity: int = Field(description="Number of catalog units (a multi-pack SKU counts as one unit).")
    reason: str = Field(description="One short sentence explaining why this SKU fits the request.")
    category: str | None = Field(default=None, description="Product category string (e.g. pos-hardware, mobile-hardware).")
    max_unit_price: float | None = Field(default=None, description="Optional unit price ceiling stated by user for this item.")
    required_specs: dict[str, Any] = Field(default_factory=dict, description="Hard spec constraints explicitly requested by user (e.g. ram_gb, has_nfc, supports_2d).")
    preferred_specs: dict[str, Any] = Field(default_factory=dict, description="Soft preferences requested by user.")


class LLMProcurementPlan(BaseModel):
    line_items: list[LLMLineItem] = Field(description="Catalog SKUs to purchase. Empty if nothing matches.")
    stated_budget: float | None = Field(
        description="Spending ceiling the requester stated, in USD, or null if none was stated."
    )
    unmatched_requests: list[str] = Field(
        description="Parts of the request that no catalog item satisfies. Empty list if all matched."
    )
    reasoning: str = Field(description="Two to four sentences explaining the trade-offs made.")


# ---------------------------------------------------------------------------
# Policy engine output
# ---------------------------------------------------------------------------


class Decision(StrEnum):
    AUTO_APPROVED = "auto_approved"
    NEEDS_APPROVAL = "needs_approval"
    BLOCKED = "blocked"


class Severity(StrEnum):
    BLOCK = "block"  # failing this check blocks the purchase outright
    REVIEW = "review"  # failing this check routes the purchase to a human


class PolicyCheck(BaseModel):
    rule: str
    passed: bool
    severity: Severity
    detail: str


class PricedLine(BaseModel):
    sku: str
    name: str
    vendor: str
    unit_price: Decimal
    quantity: int
    line_total: Decimal


class PolicyResult(BaseModel):
    decision: Decision
    checks: list[PolicyCheck]
    lines: list[PricedLine]
    subtotal: Decimal
    currency: str
    effective_budget: Decimal | None = Field(
        description="Strictest requester budget found (regex over the text and/or the LLM's reading)."
    )

    @property
    def failures(self) -> list[PolicyCheck]:
        return [c for c in self.checks if not c.passed]


# ---------------------------------------------------------------------------
# API / pipeline records
# ---------------------------------------------------------------------------


class ProcurementRequest(BaseModel):
    request: str = Field(min_length=3, max_length=2000, description="Natural-language purchase request.")


class ProcurementStatus(StrEnum):
    BLOCKED = "blocked"
    AWAITING_APPROVAL = "awaiting_approval"
    PAID = "paid"
    PAYMENT_PENDING = "payment_pending"  # PayPal accepted the capture but it hasn't settled yet
    PAYMENT_FAILED = "payment_failed"  # definitively no money moved; budget released
    NEEDS_REVIEW = "needs_review"  # money moved (or may have) unexpectedly; budget kept, finance must check
    CANCELLED = "cancelled"


class PaymentInfo(BaseModel):
    mode: Literal["vault_auto_capture", "manager_approval"]
    paypal_order_id: str
    order_status: str
    amount: Decimal
    currency: str
    approval_url: str | None = None
    capture_id: str | None = None
    capture_status: str | None = None
    note: str | None = None


class PlanAttempt(BaseModel):
    """One iteration of the plan -> policy loop (the agent's visible 'thought process')."""

    attempt: int
    feedback: str | None = Field(description="Policy feedback the agent was given before this attempt.")
    model: str | None = None
    plan: LLMProcurementPlan
    policy: PolicyResult


class ProcurementRecord(BaseModel):
    request_id: str
    created_at: datetime
    request_text: str
    status: ProcurementStatus
    plan: LLMProcurementPlan  # final plan
    policy: PolicyResult  # final policy evaluation
    attempts: list[PlanAttempt] = Field(default_factory=list)
    payment: PaymentInfo | None = None
    exemption_request: str | None = None
    error: str | None = None
