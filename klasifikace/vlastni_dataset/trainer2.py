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
MIN_LR = 1e-30
NUM_WORKERS = 4
DEVICE = "cpu"
MODEL_FILE = "model2_2.pt"

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

"""
train_ds = datasets.KMNIST(
    root="./data", train=True, download=True, transform=transform
)
test_ds = datasets.KMNIST(
    root="./data", train=False, download=True, transform=transform
)
"""

train_loader = DataLoader(
    train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS
)
test_loader = DataLoader(
    test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
)

# ---------- model ----------
class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 18, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(18, 74, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(74, 148, 5, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        self.classifier = nn.Linear(148, 10)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

model = SmallCNN().to(DEVICE)

if exists(MODEL_FILE):
    model.load_state_dict(torch.load(MODEL_FILE, map_location=DEVICE))
    model.eval()

# ---------- loss + optimizer ----------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="max",
    factor=0.5,
    patience=3,
    min_lr=MIN_LR
)

# ---------- train ----------
def train_epoch(model, loader):
    model.train()
    total_loss = 0
    correct = 0

    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)

        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        correct += (out.argmax(1) == y).sum().item()

    return total_loss / len(loader), correct / len(loader.dataset)

# ---------- test ----------
def eval_epoch(model, loader):
    model.eval()
    correct = 0

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            out = model(x)
            correct += (out.argmax(1) == y).sum().item()

    return correct / len(loader.dataset)

# ---------- main loop ----------
acc_bez_zlepseni = []
try:
    epoch = 1
    while True:
        loss, acc = train_epoch(model, train_loader)
        test_acc = eval_epoch(model, test_loader)

        scheduler.step(test_acc)

        lr = scheduler.get_last_lr()[0]
        print(
            f"Epoch {epoch:02d} | "
            f"loss {loss:.4f} | "
            f"train acc {acc:.3f} | "
            f"test acc {test_acc:.3f} | "
            f"lr {lr}"
        )

        if len(acc_bez_zlepseni) == 0 or acc - acc_bez_zlepseni[0] <= 1e-3:
            if len(acc_bez_zlepseni) > 4 and lr*0.5 <= MIN_LR:
                break
            acc_bez_zlepseni.append(acc)
        else:
            acc_bez_zlepseni[:] = [acc]

        if acc >= 0.95 or test_acc >= 0.95:
            break
        epoch += 1

finally:
    torch.save(model.state_dict(), MODEL_FILE)

print("Done.")
