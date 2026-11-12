"""Live end-to-end smoke test: real Gemini planner + policy engine + PayPal Sandbox.

Usage:  py -3.14 scripts/smoke_gemini.py ["custom request" ...]
"""

from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx  # noqa: E402

from agent import AgentError  # noqa: E402
from config import get_settings  # noqa: E402
from main import build_service  # noqa: E402

DEFAULT_REQUESTS = [
    "Restock packaging tape and one corded barcode scanner for the warehouse",
    "Procure 2 wireless barcode scanners and 24 thermal receipt rolls under $300",
    "We need a label printer and the cheapest barcode scanner you can find",
    "Order 3 coffee machines for the break room",
    "Order 1 warehouse phone with at least 8GB RAM and NFC under $150",
    "Order 1 corded USB scanner that reads QR codes",
]


async def main(requests: list[str]) -> int:
    settings = get_settings()
    failures = 0
    async with httpx.AsyncClient(timeout=settings.paypal_timeout_seconds) as http:
        service = build_service(settings, http)
        for text in requests:
            print(f"\n> {text}")
            t0 = time.perf_counter()
            try:
                rec = await service.procure(text)
            except AgentError as exc:
                failures += 1
                print(f"  [FAIL] agent error: {exc}")
                continue
            plan, pol = rec.plan, rec.policy
            print(f"  {time.perf_counter() - t0:.1f}s  decision={pol.decision}  subtotal=${pol.subtotal}  "
                  f"status={rec.status}  budget={pol.effective_budget}  attempts={len(rec.attempts)}")
            for a in rec.attempts[:-1]:
                items = ", ".join(f"{li.quantity} x {li.sku}" for li in a.plan.line_items)
                print(f"    attempt {a.attempt} [{a.model}] {items} -> {a.policy.decision}: "
                      + "; ".join(c.rule for c in a.policy.failures))
            for li in plan.line_items:
                print(f"    LLM: {li.quantity} x {li.sku} - {li.reason}")
            if plan.unmatched_requests:
                print(f"    LLM unmatched: {plan.unmatched_requests}")
            print(f"    LLM reasoning: {plan.reasoning}")
            for c in pol.failures:
                print(f"    x [{c.rule}] {c.detail}")
            if rec.payment:
                print(f"    PayPal {rec.payment.paypal_order_id} {rec.payment.order_status} {rec.payment.approval_url or ''}")
            if rec.status == "payment_failed":
                failures += 1
                print(f"  [FAIL] {rec.error}")
        print("\nbudget:", service.ledger.snapshot())
    print("RESULT:", "PASS" if not failures else f"FAIL ({failures})")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main(sys.argv[1:] or DEFAULT_REQUESTS)))
