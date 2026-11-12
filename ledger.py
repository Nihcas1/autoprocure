"""In-memory monthly spend ledger with reservations.

Approved-but-unpaid orders reserve budget so two concurrent requests cannot
both spend the last of it. Callers must hold `ledger.lock` across
"read remaining -> evaluate policy -> reserve".
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from datetime import UTC, datetime
from decimal import Decimal

from money import to_money


class SpendLedger:
    def __init__(self, monthly_budget: Decimal, clock: Callable[[], datetime] = lambda: datetime.now(UTC)):
        self.monthly_budget = to_money(monthly_budget)
        self.lock = asyncio.Lock()
        self._clock = clock
        self._period = self._current_period()
        self._committed = Decimal("0.00")
        self._pending: dict[str, Decimal] = {}

    def _current_period(self) -> str:
        return self._clock().strftime("%Y-%m")

    def _roll_period(self) -> None:
        period = self._current_period()
        if period != self._period:
            self._period = period
            self._committed = Decimal("0.00")

    def remaining(self) -> Decimal:
        self._roll_period()
        return to_money(self.monthly_budget - self._committed - sum(self._pending.values(), Decimal("0")))

    def reserve(self, key: str, amount: Decimal) -> None:
        if key in self._pending:
            raise ValueError(f"reservation already exists: {key}")
        self._pending[key] = to_money(amount)

    def commit(self, key: str, amount: Decimal | None = None) -> None:
        """Turn a reservation into committed spend. Idempotent.

        `amount` overrides the reserved figure when the processor actually took a different sum;
        the ledger must reflect real money out, not what we expected.
        """
        self._roll_period()
        reserved = self._pending.pop(key, None)
        if reserved is not None:
            self._committed = to_money(self._committed + (reserved if amount is None else amount))

    def release(self, key: str) -> None:
        """Drop a reservation (payment failed or was cancelled). Idempotent."""
        self._pending.pop(key, None)

    def snapshot(self) -> dict[str, str]:
        remaining = self.remaining()
        return {
            "period": self._period,
            "monthly_budget": f"{self.monthly_budget:.2f}",
            "committed": f"{self._committed:.2f}",
            "reserved": f"{to_money(sum(self._pending.values(), Decimal('0'))):.2f}",
            "remaining": f"{remaining:.2f}",
        }
