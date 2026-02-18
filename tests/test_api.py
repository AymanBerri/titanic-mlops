"""
API tests for Titanic Survival Prediction service.
Tests both health check and prediction endpoints.
"""

import time

import requests

BASE_URL = "http://localhost:8000"


def test_health_endpoint():
    """Test the health check endpoint returns correct status."""
    response = requests.get(f"{BASE_URL}/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "model_loaded" in data
    print("✓ Health check passed")
    return True


def test_prediction_endpoint():
    """Test the prediction endpoint with sample passenger data."""
    # Sample passenger (based on real Titanic data)
    passenger = {
        "pclass": 3,
        "sex": "male",
        "age": 22.0,
        "sibsp": 1,
        "parch": 0,
        "fare": 7.25,
        "embarked": "S",
    }

    response = requests.post(f"{BASE_URL}/predict", json=passenger)
    assert response.status_code == 200
    data = response.json()

    # Validate response structure
    assert "prediction" in data
    assert "probability" in data
    assert "survival_status" in data
    assert isinstance(data["prediction"], int)
    assert isinstance(data["probability"], float)
    assert data["survival_status"] in ["Survived", "Did not survive"]

    print(f"✓ Prediction test passed: {data}")
    return True


if __name__ == "__main__":
    print("Testing Titanic Survival Prediction API...")
    print("-" * 50)

    # Wait a moment for API to be ready
    time.sleep(1)

    # Run tests
    tests_passed = 0
    tests_total = 2

    try:
        if test_health_endpoint():
            tests_passed += 1
    except Exception as e:
        print(f"✗ Health test failed: {e}")

    try:
        if test_prediction_endpoint():
            tests_passed += 1
    except Exception as e:
        print(f"✗ Prediction test failed: {e}")

    print("-" * 50)
    print(f"Tests passed: {tests_passed}/{tests_total}")

    if tests_passed == tests_total:
        print("✅ All tests passed!")
    else:
        print("❌ Some tests failed.")
