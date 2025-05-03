from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Define transforms
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
])

# Load datasets
train_dataset = datasets.ImageFolder(root="data/train", transform=transform)
valid_dataset = datasets.ImageFolder(root="data/valid", transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False)

# Print a few labels
for imgs, labels in train_loader:
    print("Labels:", labels)  # 0 = indoor, 1 = outdoor (check class_to_idx)
    break

# Optional: print label names
print("Class mapping:", train_dataset.class_to_idx)
