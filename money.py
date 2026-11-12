"""Exact currency helpers. All money in AutoProcure is `Decimal`, never float."""

from __future__ import annotations

from decimal import ROUND_HALF_UP, Decimal, InvalidOperation

CENT = Decimal("0.01")


def to_money(value: Decimal | int | float | str) -> Decimal:
    """Convert to a 2-decimal Decimal. Floats go through `str` to avoid binary artifacts."""
    try:
        d = value if isinstance(value, Decimal) else Decimal(str(value))
    except InvalidOperation as exc:
        raise ValueError(f"not a valid money amount: {value!r}") from exc
    if not d.is_finite():
        raise ValueError(f"not a finite money amount: {value!r}")
    return d.quantize(CENT, rounding=ROUND_HALF_UP)


def fmt(amount: Decimal) -> str:
    """PayPal wire format: '123.45'."""
    return f"{to_money(amount):.2f}"
