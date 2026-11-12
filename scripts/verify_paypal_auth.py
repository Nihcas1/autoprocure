"""Manual check: can we get a PayPal OAuth token with the credentials in .env?

Usage:  py -3.14 scripts/verify_paypal_auth.py
(Moved from the repo root `test_auth.py` so pytest no longer collects it and makes live calls.)
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx  # noqa: E402

from config import get_settings  # noqa: E402
from paypal_client import PayPalClient, PayPalError  # noqa: E402


async def main() -> int:
    s = get_settings()
    cid = s.paypal_client_id.get_secret_value()
    print(f"--- PayPal {s.paypal_mode} authentication ---")
    print(f"Client ID: {cid[:8]}...{cid[-4:]}")
    print(f"Endpoint:  {s.paypal_base_url}/v1/oauth2/token")
    async with httpx.AsyncClient(timeout=15.0) as http:
        client = PayPalClient(client_id=cid, client_secret=s.paypal_client_secret.get_secret_value(),
                              base_url=s.paypal_base_url, http=http)
        try:
            await client.verify_credentials()
        except PayPalError as exc:
            print(f"[FAILED] {exc} (status={exc.status_code}, debug_id={exc.debug_id})")
            return 1
    print("[SUCCESS] Authentication successful.")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
