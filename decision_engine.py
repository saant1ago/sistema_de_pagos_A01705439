import argparse
import pandas as pd
from typing import Dict, Any, List, Tuple

DECISION_ACCEPTED = "ACCEPTED"
DECISION_IN_REVIEW = "IN_REVIEW"
DECISION_REJECTED = "REJECTED"

DEFAULT_CONFIG = {
    "amount_thresholds": {
        "digital": 2500,
        "physical": 6000,
        "subscription": 1500,
        "_default": 4000
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
    "score_to_decision": {
        "reject_at": 10,
        "review_at": 4
    }
}

# Optional: override thresholds via environment variables (for CI/CD / canary tuning)
try:
    import os as _os
    _rej = _os.getenv("REJECT_AT")
    _rev = _os.getenv("REVIEW_AT")
    if _rej is not None:
        DEFAULT_CONFIG["score_to_decision"]["reject_at"] = int(_rej)
    if _rev is not None:
        DEFAULT_CONFIG["score_to_decision"]["review_at"] = int(_rev)
except Exception:
    pass

def is_night(hour: int) -> bool:
    return hour >= 22 or hour <= 5

def high_amount(amount: float, product_type: str, thresholds: Dict[str, Any]) -> bool:
    t = thresholds.get(product_type, thresholds.get("_default"))
    return amount >= t


# ---------- Normalizadores reutilizables ----------
def _s(x: Any, default: str = "", lower: bool = True) -> str:
    s = str(x if x is not None else default)
    return s.lower() if lower else s

def _i(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return default

def _f(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default

# ---------- Contexto de score/reasons ----------
class _ScoreCtx:
    def __init__(self) -> None:
        self.score = 0
        self.reasons: List[str] = []

    def add(self, points: int, reason: str) -> None:
        if points:
            self.score += points
            sign = "+" if points >= 0 else ""
            self.reasons.append(f"{reason}({sign}{points})")

# ---------- Normalización de campos ----------
def _normalize_row(row: pd.Series) -> Dict[str, Any]:
    return {
        "chargeback_count": _i(row.get("chargeback_count", 0)),
        "ip_risk": _s(row.get("ip_risk", "low")),
        "email_risk": _s(row.get("email_risk", "low")),
        "device_risk": _s(row.get("device_fingerprint_risk", "low")),
        "rep": _s(row.get("user_reputation", "new")),
        "hour": _i(row.get("hour", 12)),
        "bin_c": _s(row.get("bin_country", ""), lower=False).upper(),
        "ip_c": _s(row.get("ip_country", ""), lower=False).upper(),
        "amount": _f(row.get("amount_mxn", 0.0)),
        "ptype": _s(row.get("product_type", "_default")),
        "lat": _i(row.get("latency_ms", 0)),
        "freq": _i(row.get("customer_txn_30d", 0)),
    }

# ---------- Reglas (cada una reduce ramas en assess_row) ----------
def _apply_hard_block(v: Dict[str, Any], cfg: Dict[str, Any], ctx: _ScoreCtx) -> bool:
    if v["chargeback_count"] >= cfg["chargeback_hard_block"] and v["ip_risk"] == "high":
        ctx.reasons.append("hard_block:chargebacks>=2+ip_high")
        return True
    return False

def _apply_categorical(v: Dict[str, Any], sw: Dict[str, Any], ctx: _ScoreCtx) -> None:
    for key, value in (
        ("ip_risk", v["ip_risk"]),
        ("email_risk", v["email_risk"]),
        ("device_fingerprint_risk", v["device_risk"]),
    ):
        pts = sw[key].get(value, 0)
        if pts:
            ctx.add(pts, f"{key}:{value}")

def _apply_reputation(v: Dict[str, Any], sw: Dict[str, Any], ctx: _ScoreCtx) -> None:
    rep_pts = sw["user_reputation"].get(v["rep"], 0)
    if rep_pts:
        # Misma forma exacta que el original (incluye + para >=0)
        sign = "+" if rep_pts >= 0 else ""
        ctx.reasons.append(f"user_reputation:{v['rep']}({sign}{rep_pts})")
        ctx.score += rep_pts

def _apply_night(v: Dict[str, Any], sw: Dict[str, Any], ctx: _ScoreCtx) -> None:
    if is_night(v["hour"]):
        ctx.add(sw["night_hour"], f"night_hour:{v['hour']}")

def _apply_geo(v: Dict[str, Any], sw: Dict[str, Any], ctx: _ScoreCtx) -> None:
    if v["bin_c"] and v["ip_c"] and v["bin_c"] != v["ip_c"]:
        ctx.add(sw["geo_mismatch"], f"geo_mismatch:{v['bin_c']}!={v['ip_c']}")

def _apply_high_amount(v: Dict[str, Any], cfg: Dict[str, Any], sw: Dict[str, Any], ctx: _ScoreCtx) -> None:
    if high_amount(v["amount"], v["ptype"], cfg["amount_thresholds"]):
        ctx.add(sw["high_amount"], f"high_amount:{v['ptype']}:{v['amount']}")
        if v["rep"] == "new":
            ctx.add(sw["new_user_high_amount"], "new_user_high_amount")

def _apply_latency(v: Dict[str, Any], cfg: Dict[str, Any], sw: Dict[str, Any], ctx: _ScoreCtx) -> None:
    if v["lat"] >= cfg["latency_ms_extreme"]:
        ctx.add(sw["latency_extreme"], f"latency_extreme:{v['lat']}ms")

def _apply_frequency_buffer(v: Dict[str, Any], ctx: _ScoreCtx) -> None:
    if v["rep"] in {"recurrent", "trusted"} and v["freq"] >= 3 and ctx.score > 0:
        ctx.add(-1, "frequency_buffer")

def _map_decision(score: int, thresholds: Dict[str, int]) -> str:
    if score >= thresholds["reject_at"]:
        return DECISION_REJECTED
    if score >= thresholds["review_at"]:
        return DECISION_IN_REVIEW
    return DECISION_ACCEPTED

# ---------- Función pública (misma firma/salida) ----------
def assess_row(row: pd.Series, cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Evalúa una transacción y devuelve decisión, score y razones.
    Refactor para reducir Cognitive Complexity moviendo reglas a helpers.
    """
    sw = cfg["score_weights"]
    v = _normalize_row(row)
    ctx = _ScoreCtx()

    # Early exit: hard block
    if _apply_hard_block(v, cfg, ctx):
        return {"decision": DECISION_REJECTED, "risk_score": 100, "reasons": ";".join(ctx.reasons)}

    # Resto de reglas
    _apply_categorical(v, sw, ctx)
    _apply_reputation(v, sw, ctx)
    _apply_night(v, sw, ctx)
    _apply_geo(v, sw, ctx)
    _apply_high_amount(v, cfg, sw, ctx)
    _apply_latency(v, cfg, sw, ctx)
    _apply_frequency_buffer(v, ctx)

    decision = _map_decision(int(ctx.score), cfg["score_to_decision"])
    return {"decision": decision, "risk_score": int(ctx.score), "reasons": ";".join(ctx.reasons)}



def run(input_csv: str, output_csv: str, config: Dict[str, Any] = None) -> pd.DataFrame:
    cfg = config or DEFAULT_CONFIG
    df = pd.read_csv(input_csv)
    results = []
    for _, row in df.iterrows():
        res = assess_row(row, cfg)
        results.append(res)
    out = df.copy()
    out["decision"] = [r["decision"] for r in results]
    out["risk_score"] = [r["risk_score"] for r in results]
    out["reasons"] = [r["reasons"] for r in results]
    out.to_csv(output_csv, index=False)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=False, default="transactions_examples.csv", help="Path to input CSV")
    ap.add_argument("--output", required=False, default="decisions.csv", help="Path to output CSV")
    args = ap.parse_args()
    out = run(args.input, args.output)
    print(out.head().to_string(index=False))

if __name__ == "__main__":
    main()
