import cv2
import os
from tqdm import tqdm

def extract(video_dir, out_dir, every=5):
    os.makedirs(out_dir, exist_ok=True)
    for vid in os.listdir(video_dir):
        if not vid.lower().endswith(('.mp4','.avi','.mov','.mkv')):
            continue
        cap = cv2.VideoCapture(os.path.join(video_dir, vid))
        count = 0
        saved = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if count % every == 0:
                cv2.imwrite(os.path.join(out_dir, f"{vid}_{saved}.jpg"), frame)
                saved += 1
            count += 1
        cap.release()

print("Extracting REAL frames...")
extract("celeb-real", "frames/real")

print("Extracting FAKE frames...")
extract("celeb-synthesis", "frames/fake")

print("DONE")