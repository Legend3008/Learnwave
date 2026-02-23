import requests
import json
import numpy as np
import os
import pandas as pd
import joblib
from sklearn.metrics.pairwise import cosine_similarity
def create_embedding(text):
    r = requests.post("http://localhost:11434/api/embeddings", json={
      "model": "bge-m3",
      "prompt": text
      })
    
    emdedding = r.json()['embedding']
    return emdedding

jsons = os.listdir("jsons")
my_list = []
chunk_id = 0
for json_file in jsons:
  with open(f"jsons/{json_file}") as f:
    content = json.load(f)
  print(f"creating embedding for {json_file}")

  for chunk in content['chunks']:
     # print(chunk)
     chunk['chunk_id'] = chunk_id
     chunk['embedding'] = create_embedding(chunk['text'])
     chunk_id += 1
     my_list.append(chunk)
  

# print(my_list)
# a = create_embedding("Hello, how are you?")
# print(a)

df = pd.DataFrame.from_records(my_list)
# print(df)
joblib.dump(df, "embeddings.joblib")
