"""Procurement pipeline: plan (LLM) -> evaluate (policy) -> execute (PayPal)."""

from __future__ import annotations

import asyncio
import logging
import uuid
from datetime import UTC, datetime

from agent import AgentError, IntentParser
from catalog import Catalog
from ledger import SpendLedger
from money import fmt, to_money
from paypal_client import PayPalClient, PayPalError, PayPalOrder
from policy import PolicyConfig, draft_exemption_request, evaluate, is_correctable, replan_feedback
from schemas import (
    Decision,
    PaymentInfo,
    PlanAttempt,
    PolicyResult,
    ProcurementRecord,
    ProcurementStatus,
)

log = logging.getLogger("autoprocure.service")


class NotFoundError(Exception):
    pass


class ProcurementService:
    def __init__(self, *, parser: IntentParser, catalog: Catalog, policy: PolicyConfig,
                 ledger: SpendLedger, paypal: PayPalClient, public_base_url: str,
                 currency: str = "USD", vault_id: str | None = None, max_replans: int = 1):
        if not 0 <= max_replans <= 2:
            raise ValueError("max_replans must be between 0 and 2")
        self.max_replans = max_replans
        self.parser = parser
        self.catalog = catalog
        self.policy = policy
        self.ledger = ledger
        self.paypal = paypal
        self.public_base_url = public_base_url.rstrip("/")
        self.currency = currency
        self.vault_id = vault_id
        self.records: dict[str, ProcurementRecord] = {}
        self._order_to_request: dict[str, str] = {}
        self._pending_setup_tokens: set[str] = set()
        self._record_locks: dict[str, asyncio.Lock] = {}

    # ---------------------------------------------------------------- lookup
    def get(self, request_id: str) -> ProcurementRecord:
        try:
            return self.records[request_id]
        except KeyError:
            raise NotFoundError(f"unknown request id: {request_id}") from None

    def get_by_order(self, order_id: str) -> ProcurementRecord:
        request_id = self._order_to_request.get(order_id)
        if request_id is None:
            raise NotFoundError(f"unknown PayPal order id: {order_id}")
        return self.records[request_id]

    # ----------------------------------------------------------------- flow
    async def procure(self, request_text: str) -> ProcurementRecord:
        request_id = f"AP-{uuid.uuid4().hex[:12].upper()}"
        attempts: list[PlanAttempt] = []
        feedback: str | None = None
        max_attempts = 1 + self.max_replans

        # plan -> policy -> (rejected & correctable? feed violations back and re-plan)
        for n in range(1, max_attempts + 1):
            try:
                plan = await self.parser.parse(request_text, self.catalog, feedback=feedback)
            except AgentError:
                if not attempts:
                    raise  # first plan failed: nothing to fall back to
                log.warning("%s re-plan %d failed; keeping the blocked plan", request_id, n, exc_info=True)
                break

            # The LLM call above runs outside the lock. Read budget -> evaluate -> reserve is
            # atomic so concurrent requests can't both spend the same money.
            async with self.ledger.lock:
                result = evaluate(plan, request_text=request_text, catalog=self.catalog, config=self.policy,
                                  monthly_remaining=self.ledger.remaining(), currency=self.currency)
                replan = n < max_attempts and result.decision is Decision.BLOCKED and is_correctable(result)
                if not replan and result.decision is not Decision.BLOCKED:
                    self.ledger.reserve(request_id, result.subtotal)

            attempts.append(PlanAttempt(attempt=n, feedback=feedback, plan=plan, policy=result,
                                        model=getattr(self.parser, "last_model_used", None)))
            log.info("%s attempt %d decision=%s subtotal=%s", request_id, n, result.decision, result.subtotal)
            if not replan:
                break
            feedback = replan_feedback(plan, result)

        final = attempts[-1]
        plan, result = final.plan, final.policy
        record = ProcurementRecord(request_id=request_id, created_at=datetime.now(UTC),
                                   request_text=request_text, status=ProcurementStatus.BLOCKED,
                                   plan=plan, policy=result, attempts=attempts)
        self.records[request_id] = record
        self._record_locks[request_id] = asyncio.Lock()

        if result.decision is Decision.BLOCKED:
            record.exemption_request = draft_exemption_request(result, request_text, request_id)
            return record

        if result.decision is Decision.AUTO_APPROVED and self.vault_id:
            await self._auto_capture(record, result)
        else:
            note = None
            if result.decision is Decision.AUTO_APPROVED:
                note = ("Auto-approved by policy (Tier 1): no manager sign-off needed. Payment uses "
                        "standard one-click PayPal checkout because no vaulted spending mandate "
                        "is configured (PAYPAL_VAULT_ID).")
            await self._request_approval(record, result, note)
        return record

    # Ledger rule: release a reservation ONLY when PayPal definitely did not move money.
    # If money moved (or we cannot tell), keep it counted and flag the request for review.

    def _fail(self, record: ProcurementRecord, error: str) -> None:
        """Definitive failure: no money moved, so the budget reservation is released."""
        self.ledger.release(record.request_id)
        record.status = ProcurementStatus.PAYMENT_FAILED
        record.error = error
        log.error("%s payment failed: %s", record.request_id, error)

    def _flag_for_review(self, record: ProcurementRecord, error: str,
                         captured_amount: str | None = None) -> None:
        """Money moved (or may have). Never free the budget; hand it to finance."""
        if captured_amount is not None:
            self.ledger.commit(record.request_id, amount=to_money(captured_amount))
        record.status = ProcurementStatus.NEEDS_REVIEW
        record.error = error
        log.critical("%s needs finance review: %s", record.request_id, error)

    @staticmethod
    def _outcome_unknown(exc: PayPalError) -> bool:
        """Network errors and 5xx may have been processed by PayPal anyway; 4xx were rejected."""
        return exc.status_code is None or exc.status_code >= 500

    async def _auto_capture(self, record: ProcurementRecord, result: PolicyResult) -> None:
        # A vaulted order is usually captured as part of creation, so this call can move money.
        try:
            order = await self._create_order(record, result,
                                             self.paypal.vault_payment_source(self.vault_id or ""))
            if order.status in ("CREATED", "APPROVED"):
                order = await self.paypal.capture_order(order.id, request_id=f"{record.request_id}-capture")
        except PayPalError as exc:
            if self._outcome_unknown(exc):
                return self._flag_for_review(record, f"Payment outcome unknown, verify in PayPal: {exc}")
            return self._fail(record, str(exc))
        record.payment = self._payment_info("vault_auto_capture", order, result)
        self._settle(record, order, result)

    async def _request_approval(self, record: ProcurementRecord, result: PolicyResult,
                                note: str | None) -> None:
        # Creating a redirect order never moves money, so any failure here is safe to release.
        source = self.paypal.approval_payment_source(
            return_url=f"{self.public_base_url}/paypal/return",
            cancel_url=f"{self.public_base_url}/paypal/cancel",
        )
        try:
            order = await self._create_order(record, result, source)
        except PayPalError as exc:
            return self._fail(record, str(exc))
        if mismatch := self._mismatch(order, result):
            return self._fail(record, mismatch)
        if not order.approval_url:
            return self._fail(record, f"PayPal order {order.id} has no approval link (status {order.status})")
        record.payment = self._payment_info("manager_approval", order, result, note=note)
        record.status = ProcurementStatus.AWAITING_APPROVAL

    async def _create_order(self, record: ProcurementRecord, result: PolicyResult,
                            payment_source: dict) -> PayPalOrder:
        order = await self.paypal.create_order(
            lines=result.lines, total=fmt(result.subtotal), currency=result.currency,
            reference_id=record.request_id,
            description=f"AutoProcure {record.request_id}: {len(result.lines)} line item(s)",
            payment_source=payment_source, request_id=f"{record.request_id}-create",
        )
        self._order_to_request[order.id] = record.request_id
        return order

    @staticmethod
    def _mismatch(order: PayPalOrder, result: PolicyResult) -> str | None:
        """Reconciliation: PayPal's echoed amount must equal the policy-approved amount."""
        if order.amount is None:
            return None
        if to_money(order.amount) != result.subtotal or (order.currency and order.currency != result.currency):
            return (f"PayPal order {order.id} amount {order.amount} {order.currency} does not match "
                    f"approved {fmt(result.subtotal)} {result.currency}")
        return None

    def _settle(self, record: ProcurementRecord, order: PayPalOrder, result: PolicyResult) -> None:
        """Apply a capture response, reconciling the amount first."""
        mismatch = self._mismatch(order, result)
        if mismatch and order.capture_status in ("COMPLETED", "PENDING"):
            # Money moved at the wrong amount: count what PayPal actually took, flag for refund/review.
            self._update_payment(record, order)
            return self._flag_for_review(record, mismatch, captured_amount=order.amount)
        if mismatch:
            return self._fail(record, mismatch)
        self._apply_capture_outcome(record, order)

    def _payment_info(self, mode, order: PayPalOrder, result: PolicyResult, note: str | None = None) -> PaymentInfo:
        return PaymentInfo(mode=mode, paypal_order_id=order.id, order_status=order.status,
                           amount=result.subtotal, currency=result.currency,
                           approval_url=order.approval_url, capture_id=order.capture_id,
                           capture_status=order.capture_status, note=note)

    @staticmethod
    def _update_payment(record: ProcurementRecord, order: PayPalOrder) -> None:
        assert record.payment is not None
        record.payment.order_status = order.status
        record.payment.capture_id = order.capture_id
        record.payment.capture_status = order.capture_status
        record.payment.approval_url = None

    def _apply_capture_outcome(self, record: ProcurementRecord, order: PayPalOrder) -> None:
        self._update_payment(record, order)
        if order.capture_status == "COMPLETED":
            self.ledger.commit(record.request_id)
            record.status = ProcurementStatus.PAID
        elif order.capture_status == "PENDING":
            # Funds are on the way; keep the budget reserved.
            record.status = ProcurementStatus.PAYMENT_PENDING
        else:
            self.ledger.release(record.request_id)
            record.status = ProcurementStatus.PAYMENT_FAILED
            record.error = f"capture status {order.capture_status or 'missing'} (order {order.status})"

    # ------------------------------------------------- manager approval loop
    async def complete_approval(self, order_id: str) -> ProcurementRecord:
        record = self.get_by_order(order_id)
        async with self._record_locks[record.request_id]:
            if record.status is not ProcurementStatus.AWAITING_APPROVAL:
                return record  # idempotent: already captured / cancelled / failed
            try:
                order = await self.paypal.capture_order(order_id, request_id=f"{record.request_id}-capture")
            except PayPalError as exc:
                if any(d.get("issue") == "ORDER_NOT_APPROVED" for d in exc.details):
                    record.error = "The PayPal order has not been approved yet."  # stay awaiting
                elif self._outcome_unknown(exc):
                    self._flag_for_review(record, f"Capture outcome unknown, verify in PayPal: {exc}")
                else:
                    self._fail(record, str(exc))
                return record
            record.error = None
            self._settle(record, order, record.policy)
            return record

    async def cancel_approval(self, order_id: str) -> ProcurementRecord:
        record = self.get_by_order(order_id)
        async with self._record_locks[record.request_id]:
            if record.status is ProcurementStatus.AWAITING_APPROVAL:
                self.ledger.release(record.request_id)
                record.status = ProcurementStatus.CANCELLED
                if record.payment:
                    record.payment.approval_url = None
            return record

    # ------------------------------------------- one-time spending mandate
    async def start_mandate_setup(self) -> str:
        token_id, approve_url = await self.paypal.create_vault_setup_token(
            return_url=f"{self.public_base_url}/mandate/return",
            cancel_url=f"{self.public_base_url}/mandate/cancel",
            request_id=f"mandate-{uuid.uuid4().hex}",
        )
        self._pending_setup_tokens.add(token_id)
        return approve_url

    async def complete_mandate_setup(self, setup_token_id: str) -> str:
        if setup_token_id not in self._pending_setup_tokens:
            raise NotFoundError("unknown or already-used setup token")
        vault_id = await self.paypal.create_payment_token(setup_token_id,
                                                          request_id=f"vault-{setup_token_id}")
        self._pending_setup_tokens.discard(setup_token_id)
        self.vault_id = vault_id
        return vault_id
