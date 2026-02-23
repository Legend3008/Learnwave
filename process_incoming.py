import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
# from read_chunks import create_embedding
import joblib
import requests

def create_embedding(text):
    r = requests.post("http://localhost:11434/api/embeddings", json={
      "model": "bge-m3",
      "prompt": text
      })
    
    emdedding = r.json()['embedding']
    return emdedding

def inference(prompt):
     r = requests.post("http://localhost:11434/api/generate", json={
      # "model":"deepseek-r1"
      "model": "llama3.2:latest",
      "prompt": prompt,
      "stream": False
      })
     response = r.json()
     print(response)
     return response

df = joblib.load("embeddings.joblib")

incoming_query = input("Ask your question: ")
question_embedding = np.array(create_embedding(incoming_query))
# print(question_embedding)

#find cosine similarity of question embedding with all chunk embeddings
# print(df['embedding'].values)
# print(df['embedding'].shape)
# df = np.array(df['embedding']).reshape(1, -1)
similarities = cosine_similarity(np.vstack(df['embedding'].apply(np.array)), question_embedding.reshape(1, -1)).flatten()
# print(similarities)
top_results = 5
max_index = similarities.argsort()[::-1][0:top_results]
# print(max_index)

new_df = df.loc[max_index]
# print(new_df[["start", "end", "text"]])

prompt = f'''I am teaching web development using Sigma web development cource. Here is some context that might be useful to answer the question:
{new_df[["start", "end", "text"]].to_json(orient="records")}
----------------------------
"{incoming_query}"
User asked the question above. Use the following context to answer the question. If you don't know the answer, just say that you don't know, don't try to make up an answer.
'''
with open("prompt.txt", "w") as f:
    f.write(prompt)

response = inference(prompt)["response"]
print(response)

with open("response.txt", "w") as f:
    f.write(response)

# for index, item in new_df.iterrows():
#     print(index, item['text'], item['start'], item['end'])