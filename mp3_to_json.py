import whisper
import json
import os
model = whisper.load_model("large-v2") 

audios = os.listdir("audios")

for audio in audios:
  if('.' in audio):
    # print(f"Processing {audio.split('.')[0]}")
    # result = model.transcribe(audio = f"audios/{audio}.mp3",
    result = model.transcribe(audio = f"audios/5.mp3",
                           language = "hi",
                           task = "translate",
                           word_timestamps=False)

    chunks = []
    for segment in result["segments"]:
        chunks.append({"start":segment["start"],          
        "end":segment["end"], 
        "text":segment["text"]})

chunks_with_metadata = {"chunks": chunks, "text": result["text"]}
with open(f"jsons/sample.json", "w") as f:
        json.dump(chunks_with_metadata, f)