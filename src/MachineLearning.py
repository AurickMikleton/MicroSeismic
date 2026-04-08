import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

data_folder = "../cv_data/"

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

transform = transforms.Compose([
    transforms.Resize((256, 256)), # Image size is already homgonize -> maybe remove
    transforms.ToTensor()
])

dataset = datasets.ImageFolder(data_folder, transform=transform)
loader = DataLoader(dataset, batch_size=32, shuffle=True)


for images, labels in loader:
    print(images.shape)   # [Batch Size, Colors, Height, Width]
    print(labels.shape)
    break


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        # 1x1 conv to match dimensions for the skip connection when shape changes
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)   # skip connection
        return self.relu(out)


class ResNet(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.prep = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )
        self.layer1 = ResidualBlock(16, 16)           # [B, 16, 256, 256]
        self.layer2 = ResidualBlock(16, 32, stride=2) # [B, 32, 128, 128]
        self.layer3 = ResidualBlock(32, 64, stride=2) # [B, 64,  64,  64]
        self.layer4 = ResidualBlock(64, 128, stride=2)# [B, 128, 32,  32]
        self.pool = nn.AdaptiveAvgPool2d(1)           # [B, 128,  1,   1]
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.prep(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


model = ResNet(num_classes=len(dataset.classes)).to(device) #now this runs on GPU or mac GPU mps ig

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)



def train_model(model, loader, optimizer, criterion, epochs=5) -> ResNet:


    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs = model(images)

            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            print(loss.item())

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {running_loss / len(loader)}")

    return model

model = train_model(model, loader, optimizer, criterion=criterion)

model.eval()
with torch.no_grad():
    images, labels = next(iter(loader))
    images, labels = images.to(device), labels.to(device)
    outputs = model(images)
    preds = torch.argmax(outputs, dim=1)

    print(dataset.class_to_idx)
    print("Predictions:", preds)
    print("Actual labels:", labels)
    print('Accuracy: ' , preds.eq(labels).sum().item() / len(labels))









