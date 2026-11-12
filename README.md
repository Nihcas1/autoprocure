# AutoProcure 🛡️💳

> **Autonomous Policy-Enforced Procurement Agent**  
> Built with Google Gemini, Python deterministic guardrails, and PayPal Sandbox financial settlement.

AutoProcure bridges the gap between natural language procurement requests and strict corporate financial compliance. While AI agents can interpret user intent and select items from catalogs, AutoProcure ensures that **zero funds move without deterministic validation against corporate policy and technical specifications**.

---

## Key Capabilities

- **Intelligent Catalog Sourcing (Gemini 2.5/Flash)**: Disentangles natural language requests into structured intent: categories, quantities, unit price limits, and hard vs. soft technical specifications.
- **Deterministic Policy Engine (Python Truth Layer)**: Evaluates 12 corporate spending rules before any payment order is generated:
  - Technical specification verification (RAM, connectivity, 2D scanning, dimensions, etc.)
  - Per-item price cap (\$150.00)
  - Per-order hard limit (\$350.00)
  - Approved vendor verification (Acme Office Supply, Globex Industrial)
  - Department monthly budget allocation and reservation ledger
- **Self-Correction Re-planning Loop**: If Gemini proposes a plan that breaches policy (e.g., an unapproved vendor or exceeding order limits), the deterministic engine feeds structured feedback back to the agent to re-plan automatically within 1 attempt.
- **Dual PayPal Settlement Flows**:
  - **Tier 1 (Autonomous Vault Mandate)**: Pre-authorized merchant-initiated capture using PayPal Vault API for auto-approved orders under \$150.
  - **Tier 2 (Manager Approval Checkout)**: Generates one-click PayPal Sandbox approval links for compliant orders that require managerial sign-off.
- **Enterprise Web Dashboard**: Real-time procurement timeline, interactive scenario presets, live policy checklist, budget tracking, and receipt copying.

---

## Architecture Overview

```
User Request (Text / UI)
         │
         ▼
[ Google Gemini AI ] ──(Plans Catalog SKUs & Specs)
         │
         ▼
[ Deterministic Policy Engine ] ──(12 Strict Python Rules)
    ├── Spec mismatch / Rule breach? ──► [ Self-Correction Feedback Loop ]
    └── Compliant?
         │
         ├── Total <= $150 ──► [ PayPal Vault Auto-Capture ] (Autonomous)
         └── Total > $150  ──► [ PayPal Sandbox Order ] (Manager Sign-Off)
```

---

## Getting Started

### 1. Prerequisites
- Python 3.14 (or Python 3.11+)
- PayPal Developer Sandbox account ([developer.paypal.com](https://developer.paypal.com))
- Google Gemini API key ([aistudio.google.com](https://aistudio.google.com))

### 2. Environment Configuration
Copy `.env.example` to `.env`:

```bash
copy .env.example .env
```

Edit `.env` with your sandbox and API credentials:

```ini
PAYPAL_CLIENT_ID=your_paypal_sandbox_client_id
PAYPAL_CLIENT_SECRET=your_paypal_sandbox_client_secret
PAYPAL_MODE=sandbox
GEMINI_API_KEY=your_gemini_api_key
GEMINI_MODEL=gemini-3.5-flash
```

*(Optional: Set `AUTOPROCURE_API_KEY` to secure endpoints with an `X-API-Key` header, or configure `PAYPAL_VAULT_ID` for pre-authorized spending mandates).*

### 3. Run the Application

Launch the FastAPI backend and dashboard:

```bash
py -3.14 -m uvicorn main:app --port 8000 --reload
```

Open your browser to:
**`http://localhost:8000`**

---

## Running the Automated Test Suite

AutoProcure includes 115 comprehensive unit and integration tests covering the agent, policy rules, PayPal client, ledger, and API endpoints:

```bash
py -3.14 -m pytest
```

---

## Repository Structure

- `main.py`: FastAPI server, lifespan events, API routes, and static dashboard mounting.
- `agent.py`: Google Gemini client, prompt engineering, structured intent extraction, and re-planning.
- `policy.py`: Deterministic corporate policy rules, specification matching, and decision engine.
- `catalog.py`: Approved vendor product catalog with structured technical specifications.
- `paypal_client.py`: Async PayPal REST v2 client (OAuth2, Orders, Captures, Vault setup).
- `ledger.py`: In-memory thread-safe budget allocation, reservation, and commitment tracker.
- `schemas.py`: Pydantic models for catalog items, LLM plans, policy checks, and payments.
- `config.py`: Environment configuration and policy thresholds using `pydantic-settings`.
- `static/index.html`: Enterprise Tailwind dashboard and audit timeline.
- `tests/`: Pytest test suite (115 tests).

---

## License

MIT License. See [LICENSE](LICENSE) for details.

