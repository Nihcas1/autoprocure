"""Live PayPal Sandbox smoke test of the full pipeline with a scripted (non-LLM) planner.

Usage:  py -3.14 scripts/smoke_live.py
Creates real sandbox orders; nothing is captured unless a vault id is configured.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx  # noqa: E402

from catalog import Catalog  # noqa: E402
from config import get_settings  # noqa: E402
from main import build_service  # noqa: E402
from schemas import LLMLineItem, LLMProcurementPlan  # noqa: E402

SCENARIOS: dict[str, LLMProcurementPlan] = {
    # $18.99 + $64.99 = $83.98 -> auto-approved (falls back to approval link without a mandate)
    "Restock 1 pack of packaging tape and 1 USB barcode scanner": LLMProcurementPlan(
        line_items=[LLMLineItem(sku="TAPE-6PK", quantity=1, reason="tape"),
                    LLMLineItem(sku="BC-SCAN-USB", quantity=1, reason="scanner")],
        stated_budget=None, unmatched_requests=[], reasoning="scripted"),
    # 2 x $89.00 + $42.00 = $220.00 -> manager approval
    "Procure 2 wireless barcode scanners and a case of thermal rolls under $300": LLMProcurementPlan(
        line_items=[LLMLineItem(sku="BC-SCAN-WL", quantity=2, reason="scanners"),
                    LLMLineItem(sku="TH-ROLL-12", quantity=1, reason="rolls")],
        stated_budget=300.0, unmatched_requests=[], reasoning="scripted"),
    # unapproved vendor + unit price over cap -> blocked
    "Get the cheapest scanner and a label printer": LLMProcurementPlan(
        line_items=[LLMLineItem(sku="BC-SCAN-BUDGET", quantity=1, reason="cheap"),
                    LLMLineItem(sku="LBL-PRINTER", quantity=1, reason="printer")],
        stated_budget=None, unmatched_requests=[], reasoning="scripted"),
}


class ScriptedParser:
    async def parse(self, request: str, catalog: Catalog, feedback: str | None = None) -> LLMProcurementPlan:
        return SCENARIOS[request]


async def main() -> int:
    settings = get_settings()
    ok = True
    async with httpx.AsyncClient(timeout=settings.paypal_timeout_seconds) as http:
        service = build_service(settings, http)
        service.parser = ScriptedParser()
        await service.paypal.verify_credentials()
        print("[ok] PayPal OAuth")

        for text in SCENARIOS:
            rec = await service.procure(text)
            p = rec.payment
            print(f"\n> {text}\n  decision={rec.policy.decision} subtotal={rec.policy.subtotal} status={rec.status}")
            for c in rec.policy.failures:
                print(f"  x [{c.rule}] {c.detail}")
            if p:
                print(f"  paypal order={p.paypal_order_id} order_status={p.order_status} mode={p.mode}")
                print(f"  approval_url={p.approval_url}")
                fetched = await service.paypal.get_order(p.paypal_order_id)
                match = fetched.amount == f"{rec.policy.subtotal:.2f}"
                ok &= match
                print(f"  [{'ok' if match else 'FAIL'}] PayPal GET amount {fetched.amount} matches policy subtotal")
                # Capturing before the manager approves must not mark it paid or release budget.
                if rec.status == "awaiting_approval":
                    rec = await service.complete_approval(p.paypal_order_id)
                    still = rec.status == "awaiting_approval"
                    ok &= still
                    print(f"  [{'ok' if still else 'FAIL'}] early capture handled: status={rec.status} ({rec.error})")
            if rec.exemption_request:
                print("  exemption request drafted:", rec.exemption_request.splitlines()[0])
            if rec.error and rec.status == "payment_failed":
                ok = False
                print("  [FAIL] error:", rec.error)

        print("\nbudget:", service.ledger.snapshot())
    print("\nRESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
