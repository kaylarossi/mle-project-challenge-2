import json
import requests


BASE_URL = "http://54.227.34.176:5000"  #update based on EC2 public url
DATA_PATH = "test/simple_post_data_subset.json"
END_POINT = "/predict/simple"


with open (DATA_PATH, 'r') as f:
    post_data = json.load(f)

response = requests.post(
    f"{BASE_URL}{END_POINT}", 
    json=post_data
)


print(f"Status code: {response.status_code}")
print(f"Response: {response.json()}")
