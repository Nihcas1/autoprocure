"""Mock multi-vendor product catalog.

Vendor names are fictional. Some entries are deliberately policy-violating
(unapproved vendor, unit price above the cap) so the guardrails have
something to catch during demos and tests.
"""

from __future__ import annotations

from collections.abc import Iterable
from decimal import Decimal
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class CatalogItem(BaseModel):
    model_config = ConfigDict(frozen=True)

    sku: str
    name: str = Field(max_length=127)
    vendor: str
    category: str
    unit_price: Decimal = Field(gt=0, decimal_places=2)
    description: str
    specs: dict[str, Any] = Field(default_factory=dict)


class Catalog:
    def __init__(self, items: Iterable[CatalogItem]):
        self._items: dict[str, CatalogItem] = {}
        for item in items:
            key = self.normalize(item.sku)
            if key in self._items:
                raise ValueError(f"duplicate SKU in catalog: {item.sku}")
            self._items[key] = item

    @staticmethod
    def normalize(sku: str) -> str:
        return sku.strip().upper()

    def get(self, sku: str) -> CatalogItem | None:
        return self._items.get(self.normalize(sku))

    def __contains__(self, sku: object) -> bool:
        return isinstance(sku, str) and self.normalize(sku) in self._items

    def all(self) -> list[CatalogItem]:
        return list(self._items.values())


def _item(sku: str, name: str, vendor: str, category: str, price: str, description: str,
          specs: dict[str, Any] | None = None) -> CatalogItem:
    return CatalogItem(
        sku=sku, name=name, vendor=vendor, category=category,
        unit_price=Decimal(price), description=description,
        specs=specs or {},
    )


DEFAULT_CATALOG = Catalog(
    [
        # Thermal paper
        _item("TH-ROLL-01", "Thermal receipt roll 3-1/8in x 230ft (single)", "Acme Office Supply",
              "pos-consumables", "4.50", "Single BPA-free thermal paper roll for POS receipt printers.",
              specs={"width_in": 3.125, "length_ft": 230, "rolls_per_pack": 1, "bpa_free": True}),
        _item("TH-ROLL-12", "Thermal receipt roll 3-1/8in x 230ft (12-pack)", "Globex Industrial",
              "pos-consumables", "42.00", "Case of 12 BPA-free thermal rolls. Cheaper per roll than singles.",
              specs={"width_in": 3.125, "length_ft": 230, "rolls_per_pack": 12, "bpa_free": True}),
        # Barcode scanners
        _item("BC-SCAN-USB", "Corded USB 1D/2D barcode scanner", "Acme Office Supply",
              "pos-hardware", "64.99", "Plug-and-play corded handheld scanner, reads 1D and QR codes.",
              specs={"connectivity": "usb", "supports_2d": True, "wireless_range_m": 0}),
        _item("BC-SCAN-WL", "Wireless 2D barcode scanner with charging cradle", "Globex Industrial",
              "pos-hardware", "89.00", "Bluetooth 2D scanner, 100m range, charging cradle included.",
              specs={"connectivity": "wireless", "supports_2d": True, "wireless_range_m": 100}),
        _item("BC-SCAN-2PK", "Wireless 2D barcode scanner bundle (2 units)", "Globex Industrial",
              "pos-hardware", "159.00", "Two wireless scanners sold as one bundle SKU.",
              specs={"connectivity": "wireless", "supports_2d": True, "wireless_range_m": 100, "units_per_pack": 2}),
        _item("BC-SCAN-BUDGET", "Budget USB barcode scanner", "QuickShip Wholesale",
              "pos-hardware", "29.99", "Low-cost 1D-only corded scanner from a marketplace reseller.",
              specs={"connectivity": "usb", "supports_2d": False, "wireless_range_m": 0}),
        # Packaging
        _item("TAPE-6PK", "Clear packaging tape 2in x 110yd (6-pack)", "Acme Office Supply",
              "packaging", "18.99", "Heavy-duty clear carton sealing tape, 6 rolls.",
              specs={"width_in": 2.0, "length_yd": 110, "rolls_per_pack": 6}),
        _item("TAPE-GUN", "Packaging tape dispenser gun", "Acme Office Supply",
              "packaging", "12.49", "Handheld tape gun for 2in rolls.",
              specs={"compatible_tape_width_in": 2.0}),
        _item("BOX-MED-25", "Medium shipping boxes 12x10x8in (25-pack)", "Globex Industrial",
              "packaging", "34.50", "Corrugated single-wall shipping boxes, 25 per bundle.",
              specs={"length_in": 12, "width_in": 10, "height_in": 8, "units_per_pack": 25}),
        # Printers
        _item("LBL-PRINTER", "Direct thermal shipping label printer", "Globex Industrial",
              "pos-hardware", "229.00", "4x6 label printer, USB and Ethernet.",
              specs={"label_size": "4x6", "connectivity": ["usb", "ethernet"], "direct_thermal": True}),
        # Mobile Devices
        _item("MOB-EDGE-PRO", "Moto Edge Enterprise Rugged Scanner Phone", "Globex Industrial",
              "mobile-hardware", "145.00", "Enterprise Android handheld with 8GB RAM, NFC, and 5000mAh battery.",
              specs={"ram_gb": 8, "storage_gb": 128, "has_nfc": True, "battery_mah": 5000}),
        _item("MOB-BUDGET-01", "Budget Warehouse Handheld Phone", "Acme Office Supply",
              "mobile-hardware", "119.00", "Entry-level Android phone with 4GB RAM, no NFC, 4000mAh battery.",
              specs={"ram_gb": 4, "storage_gb": 64, "has_nfc": False, "battery_mah": 4000}),
        _item("MOB-FLAGSHIP", "Ultra Rugged 5G Terminal", "Globex Industrial",
              "mobile-hardware", "229.00", "High-end terminal with 12GB RAM, NFC, 6000mAh battery.",
              specs={"ram_gb": 12, "storage_gb": 256, "has_nfc": True, "battery_mah": 6000}),
    ]
)
