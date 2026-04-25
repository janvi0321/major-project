import os, shutil, random

def split(src, train_dst, val_dst, ratio=0.8):
    os.makedirs(train_dst, exist_ok=True)
    os.makedirs(val_dst, exist_ok=True)

    files = [f for f in os.listdir(src) if f.endswith('.jpg')]
    random.shuffle(files)

    cut = int(len(files) * ratio)

    for f in files[:cut]:
        shutil.copy(os.path.join(src, f), os.path.join(train_dst, f))

    for f in files[cut:]:
        shutil.copy(os.path.join(src, f), os.path.join(val_dst, f))

split("frames/real", "data/train/real", "data/val/real")
split("frames/fake", "data/train/fake", "data/val/fake")

print("Dataset ready")
