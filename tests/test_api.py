from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "healthy"

def test_prediction():
    r = client.post("/predict", json={
        "area_sqft": 1500,
        "bedrooms": 3,
        "age_years": 5,
        "distance_km": 4
    })
    assert r.status_code == 200
    assert "predicted_price_lakh" in r.json()
