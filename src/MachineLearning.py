import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

data_folder = "../cv_data/"

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


model = nn.Sequential(
    # Not what the convolustion ares
    # But Google said this was the shit
    nn.Conv2d(3, 16, kernel_size=3, padding=1),  # [Batch Size, 3, 256, 256] -> [Batch Size, 16, 256, 256]
    nn.ReLU(),
    nn.MaxPool2d(2),                             # [Batch Size, 16, 128, 128]

    nn.Conv2d(16, 32, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2),                             # -> [Batch Size, 32, 64, 64]

    nn.Conv2d(32, 64, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2),                             # -> [Batch Size, 64, 32, 32]

    # I don't like shape anymore
    nn.Flatten(),
    nn.Linear(64 * 32 * 32, 256),
    nn.ReLU(),
    nn.Linear(256, 2)
)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 5

for epoch in range(epochs):
    model.train()
    running_loss = 0.0

    for images, labels in loader:
        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        print(loss.item())

    print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss / len(loader)}")


# test

model.eval()
with torch.no_grad():
    images, labels = next(iter(loader))
    outputs = model(images)
    preds = torch.argmax(outputs, dim=1)

    print(dataset.class_to_idx)
    print("Predictions:", preds)
    print("Actual labels:", labels)









