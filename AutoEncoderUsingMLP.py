# Implementation of Auto Encoder using Multilayer  Perceptron

import torch
import torch.nn as nn
from torchvision import datasets, transforms
import matplotlib.pyplot as plt


# Data Set, Transformation and Data Loader
transform = transforms.ToTensor()
mnist_data = datasets.MNIST(root='/home/idrbt-06/Desktop/PY_TORCH/Auto_Encoder/Data', train=True, download=True, transform=transform)
data_loader = torch.utils.data.DataLoader(dataset=mnist_data, batch_size=10, shuffle=True)


# Preparation of the Model of Auto Encoder
# Here we are using MLP to prepare encoder and decoder section of Auto Encoder
class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(28*28,130),
            nn.ReLU(),
            nn.Linear(130,70),
            nn.ReLU(),
            nn.Linear(70,20),
            nn.ReLU(),
            nn.Linear(20,6)
        )
        self.decoder = nn.Sequential(
            nn.Linear(6,20),
            nn.ReLU(),
            nn.Linear(20,70),
            nn.ReLU(),
            nn.Linear(70,130),
            nn.ReLU(),
            nn.Linear(130,28*28),
            nn.Sigmoid()
        )
    
    def forward(self,x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

model = Autoencoder()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


# Training of Auto Encoder
num_epochs = 502
original_and_reconstructed_images = []

for epoch in range(0, num_epochs, 1):
    for (images,labels) in data_loader:
        images = images.reshape(-1,28*28)
        outputs = model(images)
        loss = criterion(outputs, images)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f'Epoch:{epoch+1}, Loss:{loss.item():.4f}')
    original_and_reconstructed_images.append((epoch+1, images, outputs),)


# Display of Original and Reconstructed Images
# Here we are considering every 50 epoch
for k in range(0, num_epochs, 50):
    plt.figure(k)
    (a, b, c) = original_and_reconstructed_images[k]
    print('This is a=', a)
    print('This is the size of b=', b.shape)
    print('This is the size of c=', c.shape)
    original = b.reshape(-1, 28, 28)
    reconstructed = c.reshape(-1, 28, 28)
    original = original.detach().numpy()
    reconstructed = reconstructed.detach().numpy()
    for i in range(0, 10, 1):
        plt.subplot(2, 10, i+1)
        plt.imshow(original[i])
        plt.subplot(2, 10, 10+i+1)
        plt.imshow(reconstructed[i])
plt.show()

