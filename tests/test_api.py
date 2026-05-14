import pytest
from fastapi.testclient import TestClient
from ml.train import train_and_save
from app.main import app, MODEL_PATH, REFERENCE_PATH

if not MODEL_PATH.exists() or not REFERENCE_PATH.exists():
    train_and_save(MODEL_PATH, REFERENCE_PATH)


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_root_endpoint(client):
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "healthy"
    assert body["model_loaded"] is True


def test_predict_setosa(client):
    payload = {
        "sepal_length": 5.1,
        "sepal_width":  3.5,
        "petal_length": 1.4,
        "petal_width":  0.2,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["class_name"] == "setosa"
    assert 0.0 <= body["probability"] <= 1.0


def test_predict_virginica(client):
    payload = {
        "sepal_length": 6.3,
        "sepal_width":  2.8,
        "petal_length": 5.1,
        "petal_width":  1.5,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["class_id"] in (0, 1, 2)
    assert body["class_name"] in ("setosa", "versicolor", "virginica")
    assert 0.0 <= body["probability"] <= 1.0


def test_predict_invalid_input(client):
    payload = {"sepal_length": "not-a-number"}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_check_drift_no_drift(client):
    payload = {
        "samples": [
            [5.1, 3.5, 1.4, 0.2], [4.9, 3.0, 1.4, 0.2],
            [4.7, 3.2, 1.3, 0.2], [5.4, 3.9, 1.7, 0.4],
            [5.0, 3.6, 1.4, 0.2], [5.5, 2.5, 4.0, 1.3],
            [6.1, 2.9, 4.7, 1.4], [6.0, 3.0, 4.8, 1.8],
            [6.3, 2.5, 5.0, 1.9], [6.5, 3.0, 5.2, 2.0],
        ],
        "alpha": 0.05,
    }
    response = client.post("/check-drift", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert "drift_detected" in body
    assert "n_drifted_features" in body
    assert "per_feature" in body
    assert body["n_samples"] == 10


def test_check_drift_with_drift(client):
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
    response = client.post("/check-drift", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["drift_detected"] is True
    assert body["n_drifted_features"] > 0
