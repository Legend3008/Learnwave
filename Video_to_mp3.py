#convert videos to mp3
import os
import subprocess
files = os.listdir("videos")
print(files)
for file in files:  
    # print(file)
    file_name = file.split(".")[0]
    subprocess.run(["ffmpeg", "-i", f"videos/{file}", f"audios/{file_name}.mp3"])