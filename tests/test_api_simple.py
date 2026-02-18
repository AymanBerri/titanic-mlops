import requests

# Test data
passenger = {
    "pclass": 3,
    "sex": "male",
    "age": 22.0,
    "sibsp": 1,
    "parch": 0,
    "fare": 7.25,
    "embarked": "S",
}

# Make request
response = requests.post("http://localhost:8000/predict", json=passenger)

print(f"Status Code: {response.status_code}")
print(f"Response JSON: {response.json()}")
