import requests
import pandas as pd

BASE_URL = "http://54.227.34.176:5000"

with open ("test/simple_post_data_subset.json", 'r') as f:
    post_data = pd.read_json(f)


response = requests.post(
    f"{BASE_URL}/predict", 
    json=post_data.to_dict(orient='records')
)


print(f"Status code: {response.status_code}")
print(f"Response: {response.json()}")
