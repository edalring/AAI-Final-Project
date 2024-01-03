import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from models.vgg_model import VGG
from dataloader import MNISTDataset 

CUDA_VISIBLE_DEVICES=0
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

data_path ='processed_data'
trainloader = torch.utils.data.DataLoader(MNISTDataset(data_path=data_path, mode='val'), batch_size=16, shuffle=True)

# model
vgg_model = VGG().to(device)

# loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(vgg_model.parameters(), lr=0.001, momentum=0.9)

# trainning process
num_epochs = 5
for epoch in range(num_epochs):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        if device.type == 'cuda':
            # print('true')
            inputs = inputs.to(device)
            labels = labels.to(device)
        optimizer.zero_grad()
        # print(inputs.shape)
        # print(labels)
        outputs = vgg_model(inputs).to(device)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 200 == 199:
            print(f"Epoch [{epoch + 1}/{num_epochs}], Batch [{i + 1}/{len(trainloader)}], Loss: {running_loss / 200:.3f}")
            running_loss = 0.0

