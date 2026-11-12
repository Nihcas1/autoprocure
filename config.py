"""Application settings, loaded from environment variables and `.env`."""

from __future__ import annotations

from decimal import Decimal
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import AliasChoices, Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=BASE_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- PayPal ---------------------------------------------------------
    paypal_client_id: SecretStr
    paypal_client_secret: SecretStr
    paypal_mode: Literal["sandbox", "live"] = "sandbox"
    # Vaulted PayPal wallet the manager pre-authorized for agent (merchant-initiated) payments.
    # Obtained via POST /mandate/setup -> approve -> GET /mandate/return.
    paypal_vault_id: str | None = None
    paypal_timeout_seconds: float = Field(default=30.0, gt=0)

    # --- Gemini ---------------------------------------------------------
    gemini_api_key: SecretStr | None = Field(
        default=None, validation_alias=AliasChoices("GEMINI_API_KEY", "GOOGLE_API_KEY")
    )
    gemini_model: str = "gemini-3.5-flash"
    # Tried in order when the primary model is overloaded/unavailable (503/504/429/404).
    gemini_fallback_models: list[str] = ["gemini-3.1-flash-lite"]
    gemini_timeout_seconds: float = Field(default=30.0, gt=0)
    # Self-correction: how many times the agent may re-plan after a policy rejection.
    agent_max_replans: int = Field(default=1, ge=0, le=2)

    # --- App ------------------------------------------------------------
    # Shared API key for /procure, /requests and /mandate/setup. Unset or blank = no auth (local dev).
    autoprocure_api_key: SecretStr | None = None
    public_base_url: str = "http://localhost:8000"
    currency: str = "USD"
    brand_name: str = "AutoProcure"

    # --- Corporate spending policy (deterministic guardrails) -----------
    policy_auto_approve_limit: Decimal = Decimal("150.00")
    policy_order_hard_cap: Decimal = Decimal("350.00")
    policy_max_unit_price: Decimal = Decimal("150.00")
    policy_monthly_budget: Decimal = Decimal("1000.00")
    policy_max_quantity_per_line: int = Field(default=50, ge=1)
    policy_approved_vendors: list[str] = ["Acme Office Supply", "Globex Industrial"]

    @field_validator("public_base_url")
    @classmethod
    def _strip_trailing_slash(cls, v: str) -> str:
        return v.rstrip("/")

    @field_validator("paypal_vault_id", "autoprocure_api_key", mode="before")
    @classmethod
    def _blank_is_none(cls, v: object) -> object:
        return None if isinstance(v, str) and not v.strip() else v

    @model_validator(mode="after")
    def _check_policy(self) -> "Settings":
        if self.policy_auto_approve_limit > self.policy_order_hard_cap:
            raise ValueError("POLICY_AUTO_APPROVE_LIMIT must not exceed POLICY_ORDER_HARD_CAP")
        return self

    @property
    def paypal_base_url(self) -> str:
        if self.paypal_mode == "live":
            return "https://api-m.paypal.com"
        return "https://api-m.sandbox.paypal.com"


@lru_cache
def get_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]  # values come from env
