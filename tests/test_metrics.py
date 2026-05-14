from fastapi.testclient import TestClient
from ml.train import train_and_save
from app.main import app, MODEL_PATH, REFERENCE_PATH

if not MODEL_PATH.exists() or not REFERENCE_PATH.exists():
    train_and_save(MODEL_PATH, REFERENCE_PATH)

client = TestClient(app)


def test_metrics_endpoint_available():
    with TestClient(app) as c:
        response = c.get("/metrics")
    assert response.status_code == 200
    assert "ml_predictions_total" in response.text
    assert "ml_prediction_latency_seconds" in response.text
    assert "ml_prediction_confidence" in response.text
    assert "ml_model_loaded" in response.text


def test_predict_increments_counter():
    payload = {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2,
    }
    with TestClient(app) as c:
        before = c.get("/metrics").text
        c.post("/predict", json=payload)
        c.post("/predict", json=payload)
        after = c.get("/metrics").text

    assert 'class_name="setosa",status="success"' in after
    assert before != after


def test_drift_check_increments_counter():
    payload = {
        "samples": [
            [9.0, 8.0, 8.0, 5.0], [9.5, 7.5, 8.5, 5.5],
            [8.5, 8.5, 7.5, 4.5], [9.2, 8.2, 8.2, 5.2],
            [9.8, 7.8, 8.8, 5.8], [8.8, 8.8, 7.8, 4.8],
            [9.4, 8.4, 8.4, 5.4], [9.6, 7.6, 8.6, 5.6],
            [8.6, 8.6, 7.6, 4.6], [9.1, 8.1, 8.1, 5.1],
        ],
        "alpha": 0.05,
    }
    with TestClient(app) as c:
        before = c.get("/metrics").text
        c.post("/check-drift", json=payload)
        after = c.get("/metrics").text

    assert "ml_drift_checks_total" in after
    assert before != after
