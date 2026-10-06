from fastapi.testclient import TestClient
from app import app
import joblib

# Optional: mock the loaded model if model.pkl doesn't exist during test initialization
# For this basic test, we assume model.pkl has been created and the app loads it.

client = TestClient(app)

def test_predict_endpoint():
    payload = {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2
    }
    
    with TestClient(app) as client:
        response = client.post("/predict", json=payload)
        
        assert response.status_code == 200
        data = response.json()
        assert "species" in data
        assert data["species"] == "setosa"
    
def test_invalid_input():
    # Missing petal_width
    payload = {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4
    }
    with TestClient(app) as client:
        response = client.post("/predict", json=payload)
        assert response.status_code == 422 # Unprocessable Entity (Validation Error)

def test_predict_returns_probabilities():
    payload = {"sepal_length": 6.3, "sepal_width": 3.3, "petal_length": 6.0, "petal_width": 2.5}
    with TestClient(app) as client:
        data = client.post("/predict", json=payload).json()
        assert set(data["probabilities"]) == {"setosa", "versicolor", "virginica"}
        assert abs(sum(data["probabilities"].values()) - 1.0) < 1e-3
        assert data["confidence"] == data["probabilities"][data["species"]]

def test_samples_endpoint():
    with TestClient(app) as client:
        response = client.get("/samples")
        assert response.status_code == 200
        rows = response.json()
        assert len(rows) == 150
        assert {"sepal_length", "sepal_width", "petal_length", "petal_width", "species"} <= set(rows[0])
