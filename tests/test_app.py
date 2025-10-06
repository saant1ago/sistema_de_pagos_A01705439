"""
Tests for the FastAPI microservice (app.py) that exposes /transaction.
Requires: httpx (for TestClient), fastapi, pydantic.
"""

from fastapi.testclient import TestClient

# Import the FastAPI instance from app.py
# If your app file lives elsewhere (e.g., src/app.py), change the import to:
#   from src.app import app as fastapi_app
from app import app as fastapi_app

client = TestClient(fastapi_app)


def test_health():
    """Basic healthcheck should return status ok."""
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_config_contains_score_mapping():
    """Config endpoint should expose current rule thresholds/weights."""
    r = client.get("/config")
    assert r.status_code == 200
    payload = r.json()
    assert isinstance(payload, dict)
    assert "score_to_decision" in payload
    assert "amount_thresholds" in payload


def test_transaction_in_review_path():
    """Typical medium-risk digital transaction from NEW user at night -> IN_REVIEW."""
    body = {
        "transaction_id": 42,
        "amount_mxn": 5200.0,
        "customer_txn_30d": 1,
        "geo_state": "Nuevo León",
        "device_type": "mobile",
        "chargeback_count": 0,
        "hour": 23,
        "product_type": "digital",
        "latency_ms": 180,
        "user_reputation": "new",
        "device_fingerprint_risk": "low",
        "ip_risk": "medium",
        "email_risk": "new_domain",
        "bin_country": "MX",
        "ip_country": "MX"
    }
    r = client.post("/transaction", json=body)
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["transaction_id"] == 42
    assert data["decision"] in ("ACCEPTED", "IN_REVIEW", "REJECTED")
    # With the current defaults (reject_at=10, review_at=4), this should lean to IN_REVIEW
    # If you tuned env vars REJECT_AT/REVIEW_AT, this assertion may need adjustment.
    assert data["decision"] == "IN_REVIEW"


def test_transaction_hard_block_rejection():
    """Chargebacks>=2 with ip_risk=high should trigger hard block -> REJECTED."""
    body = {
        "transaction_id": 99,
        "amount_mxn": 300.0,
        "customer_txn_30d": 0,
        "geo_state": "Nuevo León",
        "device_type": "mobile",
        "chargeback_count": 2,
        "hour": 12,
        "product_type": "digital",
        "latency_ms": 100,
        "user_reputation": "new",
        "device_fingerprint_risk": "low",
        "ip_risk": "high",
        "email_risk": "low",
        "bin_country": "MX",
        "ip_country": "MX"
    }
    r = client.post("/transaction", json=body)
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["transaction_id"] == 99
    assert data["decision"] == "REJECTED"


def test_transaction_accept_path_low_risk():
    """Bajo riesgo, sin banderas -> ACCEPTED (cubre rama 'else')."""
    body = {
        "transaction_id": 1,
        "amount_mxn": 100.0,              # por debajo de cualquier umbral
        "customer_txn_30d": 0,
        "chargeback_count": 0,
        "hour": 14,                       # no es noche
        "product_type": "digital",
        "latency_ms": 100,
        "user_reputation": "new",         # 0 puntos
        "device_fingerprint_risk": "low", # 0
        "ip_risk": "low",                 # 0
        "email_risk": "low",              # 0
        "bin_country": "MX",
        "ip_country": "MX",
    }
    r = client.post("/transaction", json=body)
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["decision"] == "ACCEPTED"
    assert data["risk_score"] == 0
    assert data["transaction_id"] == 1


def test_transaction_geo_mismatch_and_night_and_latency():
    """Geo mismatch + noche + latencia extrema."""
    body = {
        "transaction_id": 2,
        "amount_mxn": 400.0,
        "customer_txn_30d": 0,
        "chargeback_count": 0,
        "hour": 23,                       # noche -> +1
        "product_type": "digital",
        "latency_ms": 3000,               # >= 2500 -> +2
        "user_reputation": "new",         # 0
        "device_fingerprint_risk": "low",
        "ip_risk": "low",
        "email_risk": "low",
        "bin_country": "US",
        "ip_country": "MX",               # mismatch -> +2
    }
    r = client.post("/transaction", json=body)
    assert r.status_code == 200, r.text
    data = r.json()
    # score esperado: 1 (noche) + 2 (lat) + 2 (geo) = 5 -> IN_REVIEW por defecto (review_at=4, reject_at=10)
    assert data["decision"] == "IN_REVIEW"
    assert "night_hour:23(+1)" in data["reasons"]
    assert "geo_mismatch:US!=MX(+2)" in data["reasons"]
    assert "latency_extreme:3000ms(+2)" in data["reasons"]


def test_transaction_high_amount_with_new_user_kicker():
    """High amount para tipo 'physical' + bonus por user nuevo."""
    body = {
        "transaction_id": 3,
        "amount_mxn": 6000.0,             # umbral exacto physical (>= dispara)
        "customer_txn_30d": 0,
        "chargeback_count": 0,
        "hour": 10,
        "product_type": "physical",
        "latency_ms": 180,
        "user_reputation": "new",         # activa new_user_high_amount (+2)
        "device_fingerprint_risk": "low",
        "ip_risk": "low",
        "email_risk": "low",
        "bin_country": "MX",
        "ip_country": "MX",
    }
    r = client.post("/transaction", json=body)
    assert r.status_code == 200, r.text
    data = r.json()
    # +2 (high_amount) +2 (new_user_high_amount) = 4 -> IN_REVIEW
    assert data["decision"] == "IN_REVIEW"
    assert "high_amount:physical:6000.0(+2)" in data["reasons"]
    assert "new_user_high_amount(+2)" in data["reasons"]


def test_transaction_trusted_frequency_buffer():
    """Usuario trusted con frecuencia alta aplica buffer -1 y queda en IN_REVIEW."""
    body = {
        "transaction_id": 4,
        "amount_mxn": 4500.0,             # digital threshold=2500 -> high_amount +2
        "customer_txn_30d": 5,            # >=3
        "chargeback_count": 0,
        "hour": 23,                       # noche -> +1 (necesario para llegar a IN_REVIEW)
        "product_type": "digital",        # ✅ enum válido
        "latency_ms": 2600,               # +2 (extrema)
        "user_reputation": "trusted",     # -2
        "device_fingerprint_risk": "low", # 0
        "ip_risk": "medium",              # +2
        "email_risk": "low",              # 0
        "bin_country": "MX",
        "ip_country": "MX",
    }
    r = client.post("/transaction", json=body)
    assert r.status_code == 200, r.text
    data = r.json()
    # score pre-buffer: +2 (ip) +2 (high_amount) +2 (lat) +1 (night) -2 (trusted) = 5
    # buffer -1 => 4 -> IN_REVIEW (con defaults review_at=4)
    assert "frequency_buffer(-1)" in data["reasons"]
    assert "night_hour:23(+1)" in data["reasons"]
    assert data["risk_score"] == 4
    assert data["decision"] == "IN_REVIEW"

