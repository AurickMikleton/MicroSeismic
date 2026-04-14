import os
import threading
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

data_folder = os.path.join(os.path.dirname(__file__), "..", "cv_data")

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

# Lazy globals
_dataset = None
_loader  = None
_model   = None
_training_lock = threading.Lock()

_training_state = {
    "running":       False,
    "epoch":         0,
    "total_epochs":  0,
    "loss":          None,
    "history":       [],   # list of {"epoch": int, "loss": float}
    "trained":       False,
    "error":         None,
}


# ── helpers ──────────────────────────────────────────────────────────────────

def _get_dataset_loader():
    global _dataset, _loader
    if _dataset is None:
        _dataset = datasets.ImageFolder(data_folder, transform=transform)
        _loader  = DataLoader(_dataset, batch_size=32, shuffle=True)
    return _dataset, _loader


# ── architecture ─────────────────────────────────────────────────────────────

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_channels)
        self.relu  = nn.ReLU(inplace=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return self.relu(out)


class ResNet(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.prep = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )
        self.layer1 = ResidualBlock(16, 16)
        self.layer2 = ResidualBlock(16, 32, stride=2)   # [B, 32, 128, 128]
        self.layer3 = ResidualBlock(32, 64, stride=2)   # [B, 64,  64,  64]
        self.layer4 = ResidualBlock(64, 128, stride=2)  # [B, 128, 32,  32]
        self.pool   = nn.AdaptiveAvgPool2d(1)
        self.fc     = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.prep(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


# ── public API ───────────────────────────────────────────────────────────────

def get_model() -> ResNet:
    global _model
    if _model is None:
        dataset, _ = _get_dataset_loader()
        _model = ResNet(num_classes=len(dataset.classes)).to(device)
    return _model


def get_training_state() -> dict:
    with _training_lock:
        return dict(_training_state)


def start_training(epochs: int = 10):
    """Launch background training thread. Returns False if already running."""
    with _training_lock:
        if _training_state["running"]:
            return False
        _training_state.update({
            "running":      True,
            "epoch":        0,
            "total_epochs": epochs,
            "loss":         None,
            "history":      [],
            "error":        None,
        })

    def _run():
        try:
            dataset, loader = _get_dataset_loader()
            model     = get_model()
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=0.001)

            for epoch in range(epochs):
                model.train()
                running_loss = 0.0
                for images, labels in loader:
                    images, labels = images.to(device), labels.to(device)
                    optimizer.zero_grad()
                    loss = criterion(model(images), labels)
                    loss.backward()
                    optimizer.step()
                    running_loss += loss.item()

                avg_loss = running_loss / len(loader)
                with _training_lock:
                    _training_state["epoch"] = epoch + 1
                    _training_state["loss"]  = avg_loss
                    _training_state["history"].append(
                        {"epoch": epoch + 1, "loss": avg_loss}
                    )

        except Exception as exc:
            with _training_lock:
                _training_state["error"] = str(exc)
        finally:
            with _training_lock:
                _training_state["running"] = True if _training_state.get("error") else False
                _training_state["trained"] = not bool(_training_state.get("error"))
                _training_state["running"] = False

    threading.Thread(target=_run, daemon=True).start()
    return True


def predict_image(pil_image) -> dict:
    """Return prediction dict for a PIL image."""
    dataset, _ = _get_dataset_loader()
    classes     = dataset.classes
    model       = get_model()

    tensor = transform(pil_image).unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        probs    = torch.softmax(model(tensor), dim=1)[0]
        pred_idx = int(torch.argmax(probs).item())

    return {
        "prediction":    classes[pred_idx],
        "confidence":    float(probs[pred_idx].item()),
        "probabilities": {cls: float(probs[i].item()) for i, cls in enumerate(classes)},
    }


def get_dataset_info() -> dict:
    dataset, _ = _get_dataset_loader()
    counts = {cls: 0 for cls in dataset.classes}
    for _, label in dataset.samples:
        counts[dataset.classes[label]] += 1
    return {
        "classes": dataset.classes,
        "counts":  counts,
        "total":   len(dataset),
    }