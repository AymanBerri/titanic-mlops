"""
API tests for Titanic Survival Prediction service.
Tests both health check and prediction endpoints using mocks.
"""

from unittest.mock import MagicMock, patch

import requests

BASE_URL = "http://localhost:8000"


@patch("requests.get")
def test_health_endpoint(mock_get):
    """Test the health check endpoint returns correct status."""
    # Create a mock response
    mock_response = MagicMock()
    mock_response.status_code = 200
    # fmt: off
    mock_response.json.return_value = {"status": "healthy", "model_loaded": True}
    # fmt: on
    mock_get.return_value = mock_response

    # Call the function (which uses requests.get)
    response = requests.get(f"{BASE_URL}/health")

    # Assertions
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "model_loaded" in data
    print("✓ Health check passed")
    return True


@patch("requests.post")
def test_prediction_endpoint(mock_post):
    """Test the prediction endpoint with sample passenger data using mocks."""
    # Sample passenger data
    passenger = {
        "pclass": 3,
        "sex": "male",
        "age": 22.0,
        "sibsp": 1,
        "parch": 0,
        "fare": 7.25,
        "embarked": "S",
    }

    # Create a mock response
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "prediction": 0,
        "probability": 0.25,
        "survival_status": "Did not survive",
    }
    mock_post.return_value = mock_response

    # Call the function (which uses requests.post)
    response = requests.post(f"{BASE_URL}/predict", json=passenger)

    # Assertions
    assert response.status_code == 200
    data = response.json()

    assert "prediction" in data
    assert "probability" in data
    assert "survival_status" in data
    assert isinstance(data["prediction"], int)
    assert isinstance(data["probability"], float)
    assert data["survival_status"] in ["Survived", "Did not survive"]

    print(f"✓ Prediction test passed: {data}")
    return True


if __name__ == "__main__":
    print("Testing Titanic Survival Prediction API " "(with mocks)...")
    print("-" * 50)

    # Run tests
    test_health_endpoint()
    test_prediction_endpoint()

    print("-" * 50)
    print("✅ All tests passed!")
