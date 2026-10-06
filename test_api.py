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
