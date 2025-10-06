import argparse
import os
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Tuple

import pandas as pd

# ----------------------------
# Decisions (public constants)
# ----------------------------
DECISION_ACCEPTED = "ACCEPTED"
DECISION_IN_REVIEW = "IN_REVIEW"
DECISION_REJECTED = "REJECTED"

# ----------------------------
# Default configuration
# ----------------------------
DEFAULT_CONFIG: Dict[str, Any] = {
    "amount_thresholds": {
        "digital": 2500,
        "physical": 6000,
        "subscription": 1500,
        "_default": 4000,
    },
    "latency_ms_extreme": 2500,
    "chargeback_hard_block": 2,
    "score_weights": {
        "ip_risk": {"low": 0, "medium": 2, "high": 4},
        "email_risk": {"low": 0, "medium": 1, "high": 3, "new_domain": 2},
        "device_fingerprint_risk": {"low": 0, "medium": 2, "high": 4},
        "user_reputation": {"trusted": -2, "recurrent": -1, "new": 0, "high_risk": 4},
        "night_hour": 1,
        "geo_mismatch": 2,
        "high_amount": 2,
        "latency_extreme": 2,
        "new_user_high_amount": 2,
    },
    "score_to_decision": {"reject_at": 10, "review_at": 4},
}

# Optional: override thresholds via environment variables (for CI/CD / canary tuning)
try:
    if (rej := os.getenv("REJECT_AT")) is not None:
        DEFAULT_CONFIG["score_to_decision"]["reject_at"] = int(rej)
    if (rev := os.getenv("REVIEW_AT")) is not None:
        DEFAULT_CONFIG["score_to_decision"]["review_at"] = int(rev)
except Exception:
    # Keep defaults on any parsing error
    pass


# ----------------------------
# Small utilities
# ----------------------------
def is_night(hour: int) -> bool:
    return hour >= 22 or hour <= 5


def high_amount(amount: float, product_type: str, thresholds: Dict[str, Any]) -> bool:
    t = thresholds.get(product_type, thresholds.get("_default"))
    return amount >= t


def _norm_str(x: Any, *, default: str = "", to_lower: bool = True) -> str:
    s = str(x if x is not None else default)
    return s.lower() if to_lower else s


def _norm_int(x: Any, *, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return default


def _norm_float(x: Any, *, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


# ----------------------------
# Core scoring / decisioning
# ----------------------------
@dataclass
class RiskResult:
    decision: str
    risk_score: int
    reasons: str


def assess_row(row: "pd.Series", cfg: Dict[str, Any]) -> RiskResult:
    """
    Refactor highlights:
    - Normalization helpers to avoid repeated casting/KeyErrors.
    - Table-driven categorical scoring.
    - Single `add()` helper to atomically update score & reasons with consistent formatting.
    - Early return for hard block.
    """
    score = 0
    reasons: List[str] = []
    sw = cfg["score_weights"]

    def add(points: int, reason: str) -> None:
        nonlocal score
        if points:
            score += points
            sign = "+" if points >= 0 else ""
            reasons.append(f"{reason}({sign}{points})")

    # ---- Pre-normalize all inputs once
    chargeback_count = _norm_int(row.get("chargeback_count", 0))
    ip_risk = _norm_str(row.get("ip_risk", "low"))
    email_risk = _norm_str(row.get("email_risk", "low"))
    device_risk = _norm_str(row.get("device_fingerprint_risk", "low"))
    user_rep = _norm_str(row.get("user_reputation", "new"))
    hour = _norm_int(row.get("hour", 12))
    bin_country = _norm_str(row.get("bin_country", ""), to_lower=False).upper()
    ip_country = _norm_str(row.get("ip_country", ""), to_lower=False).upper()
    amount_mxn = _norm_float(row.get("amount_mxn", 0.0))
    product_type = _norm_str(row.get("product_type", "_default"))
    latency_ms = _norm_int(row.get("latency_ms", 0))
    freq_30d = _norm_int(row.get("customer_txn_30d", 0))

    # ---- Hard block (early exit)
    if chargeback_count >= cfg["chargeback_hard_block"] and ip_risk == "high":
        reasons.append("hard_block:chargebacks>=2+ip_high")
        return RiskResult(DECISION_REJECTED, 100, ";".join(reasons))

    # ---- Categorical risks (table-driven)
    categorical_fields: List[Tuple[str, str]] = [
        ("ip_risk", ip_risk),
        ("email_risk", email_risk),
        ("device_fingerprint_risk", device_risk),
    ]
    for key, value in categorical_fields:
        add(sw[key].get(value, 0), f"{key}:{value}")

    # ---- Reputation
    add(sw["user_reputation"].get(user_rep, 0), f"user_reputation:{user_rep}")

    # ---- Night hour
    if is_night(hour):
        add(sw["night_hour"], f"night_hour:{hour}")

    # ---- Geo mismatch
    if bin_country and ip_country and bin_country != ip_country:
        add(sw["geo_mismatch"], f"geo_mismatch:{bin_country}!={ip_country}")

    # ---- High amount (+ new-user kicker)
    if high_amount(amount_mxn, product_type, cfg["amount_thresholds"]):
        add(sw["high_amount"], f"high_amount:{product_type}:{amount_mxn}")
        if user_rep == "new":
            add(sw["new_user_high_amount"], "new_user_high_amount")

    # ---- Extreme latency
    if latency_ms >= cfg["latency_ms_extreme"]:
        add(sw["latency_extreme"], f"latency_extreme:{latency_ms}ms")

    # ---- Frequency buffer for trusted/recurrent
    if user_rep in {"recurrent", "trusted"} and freq_30d >= 3 and score > 0:
        add(-1, "frequency_buffer")

    # ---- Decision mapping
    reject_at = cfg["score_to_decision"]["reject_at"]
    review_at = cfg["score_to_decision"]["review_at"]
    if score >= reject_at:
        decision = DECISION_REJECTED
    elif score >= review_at:
        decision = DECISION_IN_REVIEW
    else:
        decision = DECISION_ACCEPTED

    return RiskResult(decision=decision, risk_score=int(score), reasons=";".join(reasons))


# ----------------------------
# Batch runner
# ----------------------------
def run(input_csv: str, output_csv: str, config: Dict[str, Any] | None = None) -> pd.DataFrame:
    cfg = config or DEFAULT_CONFIG
    df = pd.read_csv(input_csv)

    # Apply row-wise (keeps memory usage simple and explicit)
    results = df.apply(lambda r: assess_row(r, cfg), axis=1)

    # Expand results into columns
    out = df.copy()
    out["decision"] = results.map(lambda rr: rr.decision)
    out["risk_score"] = results.map(lambda rr: rr.risk_score)
    out["reasons"] = results.map(lambda rr: rr.reasons)

    out.to_csv(output_csv, index=False)
    return out


# ----------------------------
# CLI
# ----------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="transactions_examples.csv", help="Path to input CSV")
    ap.add_argument("--output", default="decisions.csv", help="Path to output CSV")
    args = ap.parse_args()

    out = run(args.input, args.output)
    # Keep a concise preview for CLI usage
    print(out.head().to_string(index=False))


if __name__ == "__main__":
    main()
