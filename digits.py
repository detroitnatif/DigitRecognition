from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import torch

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
                        num_workers=1),
    'test': DataLoader(test_data,
                        batch_size=100,
                        shuffle=True,
                        num_workers=1)
}

class CNN(nn.Module):

    def __init__(self):
        super(CNN, self).__init__()

        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)  # OUTPUT IS 10 becuase there is 10 possibilities

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x)), 2)
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x))), 2)
        x.view(-1, 320)
        x.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        x = F.softmax(x)

        return F.softmax(x)
    
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CNN().to(device)

optimizer = optim.Adam(model.parameters(), lr = .01)

loss_fn = nn.CrossEntropyLoss()

def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(loaders['train']):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        ypred = model(data)
        loss = loss_fn(ypred, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 50 == 0:
            print(f"train epoch {epoch} loss [{batch_idx * len(data)}/{len(loaders['train'].dataset)} ({100. * batch_idx / len(loaders['train']):.0f}%)]\t{loss.item():.6f}")


            
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
    
for epoch in range(1, 11):
    train(epoch)
    test()