import pandas as pd
import json 

df = pd.read_csv('test/future_unseen_examples.csv')
records =df.to_dict(orient='records')

with open("test/post_data.json", "w") as f:
    json.dump(records, f)

print(f"Converted {len(records)} records to JSON")
