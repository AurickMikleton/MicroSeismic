from torchvision import datasets, transforms
from torch.utils.data import DataLoader

data_folder = "../cv_data/"

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

dataset = datasets.ImageFolder(data_folder, transform=transform)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

for images, labels in loader:
    print(images.shape)   # [B, C, H, W]
    print(labels.shape)
    break
