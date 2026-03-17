import requests
import json

BASE_URL = "http://18.207.115.247:5000"

with open("test/post_data.json", "r") as f:
    data = json.load(f)

response = requests.post(
    f"{BASE_URL}/predict", 
    json=data)
print(f"Status code: {response.status_code}")
print(f"Response: {response.json()}")
