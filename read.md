#how to use this RAS AI Teaching Assistant on your data
step 1: Collect your videos
 Move all your files to the video folder

step 2: Convert to mp3
 convert all the video files to mp3 by running video_to_mp3

step 3: Convert mp3 to json
 convert all the mp3 files to json by running mp3_to_json

step 4: Convert the json files to vectors
 Use the files preprocess_json to convert the json files to a dataframe with Embeddings and save it as a joblib pkl

step 5: Prompt generation and feeding to LLM 
 Read the joblib file and load it into the memory, then create a revelant prompt as per the user query and feed it to LLm