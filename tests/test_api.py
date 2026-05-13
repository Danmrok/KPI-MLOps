import pytest
from fastapi.testclient import TestClient
from ml.train import train_and_save
from app.main import app, MODEL_PATH

if not MODEL_PATH.exists():
    train_and_save(MODEL_PATH)


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
