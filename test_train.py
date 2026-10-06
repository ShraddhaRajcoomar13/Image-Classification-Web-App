import ssl
ssl._create_default_https_context = ssl._create_unverified_context

from cifar10cnn import CIFAR10CNN
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn as nn, torch.optim as optim

print("1. Import OK")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("2. Device:", device)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914,0.4822,0.4465),(0.2023,0.1994,0.2010))
])
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=0)
print("3. Data loaded:", len(trainset))

model = CIFAR10CNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# One tiny step to prove the loop runs
inputs, labels = next(iter(trainloader))
inputs, labels = inputs.to(device), labels.to(device)
optimizer.zero_grad()
outputs = model(inputs)
loss = criterion(outputs, labels)
loss.backward()
optimizer.step()
print("4. One training step OK, loss =", loss.item())

torch.save({'model_state_dict': model.state_dict(), 'accuracy': 0.0}, 'cifar10_cnn.pth')
print("5. Saved cifar10_cnn.pth")