import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, TensorDataset, Subset
from sys import argv
from os.path import exists

# ---------- konfigurace ----------
BATCH_SIZE = 128
EPOCHS = 45
LR = 1e-3
MIN_LR = 1e-10
NUM_WORKERS = 4
DEVICE = "cpu"
MODEL_FILE = "model_3.pt"

# ---------- dataset ----------
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

data = torch.load("cifar10_train.pt", map_location="cpu")

train_ds = TensorDataset(
    data["images"],
    data["labels"]
)

data = torch.load("cifar10_test.pt", map_location="cpu")

test_ds = TensorDataset(
    data["images"],
    data["labels"]
)

test_loader = DataLoader(
    test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
)

# ---------- model ----------
class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 18, 7, padding=1),
            nn.ReLU(),
            nn.Conv2d(18, 54, 5, padding=1),
            nn.ReLU(),
            nn.Conv2d(54, 108, 4, padding=1),
            nn.ReLU(),
            nn.Conv2d(108, 162, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        self.classifier = nn.Linear(162, 10)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

model = SmallCNN().to(DEVICE)
model.load_state_dict(torch.load(MODEL_FILE, map_location=DEVICE))

# ---------- testovací funkce ----------
criterion = nn.CrossEntropyLoss()

def eval_model(model, loader):
    num_classes = 10
    confusion = torch.zeros(num_classes, num_classes, dtype=torch.int64)
    model.eval()

    correct = 0
    total_loss = 0

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            out = model(x)
            preds = out.argmax(dim=1)
            correct += (preds == y).sum().item()
            total_loss += criterion(out, y).item()

            for t, p in zip(y.view(-1), preds.view(-1)):
                confusion[t.long(), p.long()] += 1

    return total_loss / len(loader), correct / len(loader.dataset), confusion

# ---------- test ----------
acc_bez_zlepseni = []
loss, acc, confusion = eval_model(model, test_loader)

print(
    f"LOSS:     {loss:.3f}\n"
    f"ACCURACY: {acc:.3f}\n"
)

print(confusion)
