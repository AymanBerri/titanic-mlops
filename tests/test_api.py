import time

import requests


def test_api_health():
    """Test the health endpoint"""
    try:
        response = requests.get("http://localhost:8000/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        print("✅ Health check passed")
        return True
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return False


def test_prediction():
    """Test the prediction endpoint"""
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

    try:
        response = requests.post("http://localhost:8000/predict", json=passenger)
        assert response.status_code == 200
        data = response.json()
        assert "prediction" in data
        assert "probability" in data
        assert "survival_status" in data
        print(f"✅ Prediction test passed: {data}")
        return True
    except Exception as e:
        print(f"❌ Prediction test failed: {e}")
        return False


if __name__ == "__main__":
    print("Testing Titanic Survival Prediction API...")
    print("-" * 50)

    # Wait for API to start
    time.sleep(2)

    # Run tests
    health_ok = test_api_health()
    if health_ok:
        test_prediction()

    print("-" * 50)
    print("Tests completed")
