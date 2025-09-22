import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm 

from src.datasets.cifar100 import loadData
from src.models.resnet18 import Resnet18


def trainOneEpoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, total_correct, total_samples = 0, 0, 0

    for images, labels in tqdm(loader, desc="train", ncols=80):
        images, labels = images.to(device), labels.to(device)

        logits = model(images)
        loss = criterion(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * labels.size(0)

        preds = logits.argmax(dim=1)
        total_correct += (preds == labels).sum().item()
        total_samples += labels.size(0)

    avg_loss = total_loss / total_samples
    avg_acc = 100.0 * total_correct / total_samples
    return avg_loss, avg_acc


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, total_correct, total_samples = 0, 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        logits = model(images)
        loss = criterion(logits, labels)

        total_loss += loss.item() * labels.size(0)
        preds = logits.argmax(dim=1)
        total_correct += (preds == labels).sum().item()
        total_samples += labels.size(0)

    avg_loss = total_loss / total_samples
    avg_acc = 100.0 * total_correct / total_samples
    return avg_loss, avg_acc


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("the device u're using:", device)


    train_loader, val_loader, _ = loadData(batch=128)


    model = Resnet18(classes=100).to(device)


    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)


    for epoch in range(3):
        tr_loss, tr_acc = trainOneEpoch(model, train_loader, criterion, optimizer, device)
        va_loss, va_acc = evaluate(model, val_loader, criterion, device)
        print(f"epoch {epoch+1}: "
              f"train {tr_acc:.2f}% (loss {tr_loss:.4f}) | "
              f"val {va_acc:.2f}% (loss {va_loss:.4f})")


if __name__ == "__main__":
    main()
