import os
import argparse
import cv2
import numpy as np
from scipy import fftpack
from tqdm import tqdm

# Torch
import torch
import torch.nn as nn
from torchvision import transforms, models, datasets
from torch.utils.data import DataLoader


# ---------------------------
# FRAME EXTRACTION
# ---------------------------
def extract_frames(video_path, every_n_frames=5, max_frames=None):
    cap = cv2.VideoCapture(video_path)
    frames = []
    idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if idx % every_n_frames == 0:
            frames.append((idx, frame))

        idx += 1
        if max_frames and len(frames) >= max_frames:
            break

    cap.release()
    return frames, 30


# ---------------------------
# MODEL
# ---------------------------
def make_model(num_classes=2, device="cpu"):
    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model.to(device)


# ---------------------------
# TRAIN (UPDATED WITH RESUME)
# ---------------------------
def train_classifier(data_dir, save_ckpt, epochs=10, batch_size=16, lr=1e-4):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[train] Using device: {device}")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])

    train_ds = datasets.ImageFolder(os.path.join(data_dir, "train"), transform=transform)
    val_ds   = datasets.ImageFolder(os.path.join(data_dir, "val"), transform=transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size)

    model = make_model(len(train_ds.classes), device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # ✅ RESUME
    start_epoch = 0

    if os.path.exists(save_ckpt):
        print("🔄 Resuming from checkpoint...")
        checkpoint = torch.load(save_ckpt, map_location=device)

        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        start_epoch = checkpoint["epoch"] + 1

    try:
        for epoch in range(start_epoch, epochs):

            # TRAIN
            model.train()
            total, correct = 0, 0

            for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
                imgs, labels = imgs.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += imgs.size(0)

            train_acc = correct / total

            # VALIDATION
            model.eval()
            val_correct, val_total = 0, 0

            with torch.no_grad():
                for imgs, labels in val_loader:
                    imgs, labels = imgs.to(device), labels.to(device)
                    outputs = model(imgs)
                    _, preds = torch.max(outputs, 1)
                    val_correct += (preds == labels).sum().item()
                    val_total += imgs.size(0)

            val_acc = val_correct / val_total

            print(f"Epoch {epoch+1} → Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

            # ✅ SAVE EVERY EPOCH
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "classes": train_ds.classes
            }, save_ckpt)

    except KeyboardInterrupt:
        print("\n⏸ Training paused. Saving checkpoint...")

        torch.save({
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "classes": train_ds.classes
        }, save_ckpt)

        print("✅ Saved! Run again to resume.")


# ---------------------------
# PREDICT
# ---------------------------
def predict(video_path, ckpt):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = torch.load(ckpt, map_location=device)

    classes = checkpoint["classes"]
    model = make_model(len(classes), device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    frames, _ = extract_frames(video_path)
    transform = transforms.Compose
    transforms.ToPILImage(),
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])

 
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])

    probs = []

    for _, frame in frames:
        img = transform(frame).unsqueeze(0).to(device)

        with torch.no_grad():
            out = model(img)
            p = torch.softmax(out, dim=1)[0][1].item()
            probs.append(p)

    mean_prob = np.mean(probs)
    label = "Fake" if mean_prob > 0.5 else "Real"

    print("\n=== RESULT ===")
    print("Prediction:", label)
    print("Confidence:", mean_prob)


# ---------------------------
# MAIN
# ---------------------------
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--mode", required=True, choices=["train", "predict"])
    parser.add_argument("--data_dir")
    parser.add_argument("--video")
    parser.add_argument("--ckpt", default="model.pth")

    args = parser.parse_args()

    if args.mode == "train":
        train_classifier(args.data_dir, args.ckpt)

    elif args.mode == "predict":
        predict(args.video, args.ckpt)


if __name__ == "__main__":
    main()