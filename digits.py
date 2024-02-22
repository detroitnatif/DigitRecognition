from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import torch
import numpy as np



train_data = datasets.MNIST(
    root='data',
    train = True,
    transform= ToTensor(),
    download=False
)

test_data = datasets.MNIST(
    root='data',
    train = False,
    transform= ToTensor(),
    download=False
)


inputs = train_data.data.shape  # 60_000 x 28 x 28
targets = train_data.targets    # 60_000 (what each matrix number is)

loaders = {
    'train': DataLoader(train_data,
                        batch_size=100,
                        shuffle=True,
                        num_workers=0),
    'test': DataLoader(test_data,
                        batch_size=100,
                        shuffle=True,
                        num_workers=0)
}

class CNN(nn.Module):

    def __init__(self):
        super(CNN, self).__init__()

        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)  # Kernel knocks off n-1 dims
        self.conv2 = nn.Conv2d(10, 10, kernel_size=5) 
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(160, 50)
        self.fc2 = nn.Linear(50, 10)  # OUTPUT IS 10 becuase there is 10 possibilities
    
        self.seq = nn.Sequential(
            self.conv1,
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(),
            self.conv2,
            self.conv2_drop,
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(),
            nn.Flatten(),
            self.fc1,
            nn.ReLU(),
            nn.Dropout(),  # Dropout doesn't take any arguments
            self.fc2,
            nn.Softmax(dim=1)
        )
        
    def forward(self, x):
        x = self.seq(x)
        x = F.softmax(x, dim=1) 
        return x

    # def forward(self, x):
    #     print('1', x.shape)  #          [50, 1, 28, 28])
    #     x = self.conv1(x)
    #     print('2',x.shape)
    #     x = F.relu(F.max_pool2d(x, 2))
    #     print('3', x.shape)  #          [50, 10, 12, 12])
    #     x = self.conv2_drop(self.conv2(x))
    #     x = F.relu(F.max_pool2d(x, 2))
    #     print('4', x.shape)  #          ([50, 20, 4, 4])
    #     x = self.fc1(x.view(-1, 160))
    #     print('5', x.shape)  #           ([160, 50])
    #     x = F.dropout(x, training=self.training)
    #     x = self.fc2(x)
    #     print('6', x.shape)
    #     x = F.softmax(x, dim=1)

    #     return x
    
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CNN().to(device)

optimizer = optim.Adam(model.parameters(), lr = .01)

loss_fn = nn.CrossEntropyLoss()


loss_o = 0
def train(epoch):
    model.train()
    for i in range(epoch):
        for batch_idx, (data, target) in enumerate(loaders['train']):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            ypred = model(data)
            
            loss = loss_fn(ypred, target)
            loss_o = loss
            loss.backward()
            optimizer.step()
            if batch_idx % 100 == 0:
                print(f"train epoch {i} loss [{batch_idx * len(data)}/{len(loaders['train'].dataset)} ({100. * batch_idx / len(loaders['train']):.0f}%)]\t{loss.item():.6f}")
    print('finished')
    model.eval()

            
def test():
    model.eval()

    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in loaders['test']:
            data, target = data.to(device), target.to(device)
            ypred = model(data)
            test_loss += loss_fn(ypred, target).item()
            prediction = ypred.argmax(dim=1, keepdim=True)
            correct += prediction.eq(target.view_as(prediction)).sum().item()
    test_loss /= len(loaders['test'].dataset)
    print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy {correct}/{len(loaders["test"].dataset)} ({100. * correct / len(loaders["test"].dataset):.0f}%)')

# print(loaders['train'].dataset[0])

train(1)
test()


import matplotlib.pyplot as plt

def evaluate():
    model.eval()
    idx = np.random.randint(0, len(test_data) - 1)
    print(idx)
    data, target = test_data[idx]
    # print('data', data)
    data = data.unsqueeze(0)
    print('target', target)
    output = model(data)
    print(output)
    pred = output.argmax(1, keepdim=True).item()
    print("prediction ", pred)

    image = data.squeeze(0, 1)

    plt.imshow(image)
    plt.show()

evaluate()