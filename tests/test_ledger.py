from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

import pytest

from ledger import SpendLedger


def test_reserve_commit_release():
    ledger = SpendLedger(Decimal("100.00"))
    ledger.reserve("a", Decimal("30.00"))
    ledger.reserve("b", Decimal("20.00"))
    assert ledger.remaining() == Decimal("50.00")
    ledger.commit("a")
    ledger.release("b")
    assert ledger.remaining() == Decimal("70.00")
    snap = ledger.snapshot()
    assert snap["committed"] == "30.00" and snap["reserved"] == "0.00"


def test_commit_and_release_are_idempotent():
    ledger = SpendLedger(Decimal("100.00"))
    ledger.reserve("a", Decimal("30.00"))
    ledger.commit("a")
    ledger.commit("a")
    ledger.release("a")
    assert ledger.remaining() == Decimal("70.00")


def test_double_reservation_is_rejected():
    ledger = SpendLedger(Decimal("100.00"))
    ledger.reserve("a", Decimal("1.00"))
    with pytest.raises(ValueError):
        ledger.reserve("a", Decimal("1.00"))


def test_committed_spend_resets_on_new_month():
    now = [datetime(2026, 11, 30, 23, 59, tzinfo=UTC)]
    ledger = SpendLedger(Decimal("100.00"), clock=lambda: now[0])
    ledger.reserve("a", Decimal("60.00"))
    ledger.commit("a")
    assert ledger.remaining() == Decimal("40.00")
    now[0] = datetime(2026, 12, 1, 0, 1, tzinfo=UTC)
    assert ledger.remaining() == Decimal("100.00")
