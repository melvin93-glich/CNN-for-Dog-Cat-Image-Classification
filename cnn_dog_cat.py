"""
Dog vs Cat image classification with a 3-layer CNN (PyTorch).

Expected folder layout (e.g. the Kaggle "Dogs vs Cats" dataset, split by you):
    data/
      train/
        cat/  *.jpg
        dog/  *.jpg
      val/
        cat/  *.jpg
        dog/  *.jpg

Run:  python cnn_dog_cat.py --data data --epochs 15
Predict one image:  python cnn_dog_cat.py --predict path/to/image.jpg
"""
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from PIL import Image

IMG_SIZE = 128
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CatDogCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # Three convolutional blocks: Conv -> BatchNorm -> ReLU -> MaxPool
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),    # Conv layer 1
            nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),   # 128 -> 64
            nn.Conv2d(32, 64, kernel_size=3, padding=1),   # Conv layer 2
            nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),   # 64 -> 32
            nn.Conv2d(64, 128, kernel_size=3, padding=1),  # Conv layer 3
            nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),  # 32 -> 16
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 16 * 16, 256), nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 1),  # single logit: 0 = cat, 1 = dog
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def get_loaders(root, batch_size=32):
    norm = transforms.Normalize([0.5] * 3, [0.5] * 3)
    train_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(), norm,
    ])
    val_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(), norm,
    ])
    train_ds = datasets.ImageFolder(f"{root}/train", train_tf)
    val_ds = datasets.ImageFolder(f"{root}/val", val_tf)
    print("Classes:", train_ds.class_to_idx)  # {'cat': 0, 'dog': 1}
    return (DataLoader(train_ds, batch_size, shuffle=True, num_workers=0),
            DataLoader(val_ds, batch_size, num_workers=0))


def run_epoch(model, loader, criterion, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss, correct, n = 0.0, 0, 0
    with torch.set_grad_enabled(training):
        for x, y in loader:
            x, y = x.to(device), y.float().unsqueeze(1).to(device)
            out = model(x)
            loss = criterion(out, y)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * x.size(0)
            correct += ((out > 0).float() == y).sum().item()
            n += x.size(0)
    return total_loss / n, correct / n


def train(args):
    train_loader, val_loader = get_loaders(args.data)
    model = CatDogCNN().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    best = 0.0
    for epoch in range(1, args.epochs + 1):
        tl, ta = run_epoch(model, train_loader, criterion, optimizer)
        vl, va = run_epoch(model, val_loader, criterion)
        print(f"Epoch {epoch:02d} | train loss {tl:.4f} acc {ta:.3f} "
              f"| val loss {vl:.4f} acc {va:.3f}")
        if va > best:
            best = va
            torch.save(model.state_dict(), "cat_dog_cnn.pt")
    print(f"Best validation accuracy: {best:.3f}")


def predict(path):
    model = CatDogCNN().to(device)
    model.load_state_dict(torch.load("cat_dog_cnn.pt", map_location=device))
    model.eval()
    tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3),
    ])
    x = tf(Image.open(path).convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        p = torch.sigmoid(model(x)).item()
    print(f"{'Dog' if p > 0.5 else 'Cat'} (dog probability = {p:.3f})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--predict", default=None)
    args = ap.parse_args()
    predict(args.predict) if args.predict else train(args)
