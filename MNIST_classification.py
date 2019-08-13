import torch.nn as nn
import torch.optim as optim
import mdn
import torch
from torchvision import datasets, transforms
import numpy as np

torch.set_default_tensor_type('torch.FloatTensor')

batch_size = 1000

def test(model, test_loader):

    error_sum = 0

    for batch_idx, (minibatch, labels) in enumerate(test_loader):

        minibatch = minibatch.reshape(batch_size, 1, 784)

        labels = labels.reshape(batch_size,1)
        labels = labels.int()

        pi, sigma, mu = model(minibatch)

        samples = mdn.sample(pi, sigma, mu).int()

        error = (samples != labels)

        error_sum =  error_sum +  error.sum()


    return 1 - error_sum.item()/(len(test_loader)*test_loader.batch_size)

def train(model, train_loader, num_ep):

    optimizer = optim.Adam(model.parameters(),lr=0.001)

    # train the model
    for epoch in range(num_ep):

        print(epoch)

        for batch_idx, (minibatch, labels) in enumerate(train_loader):

            minibatch = minibatch.reshape(batch_size, 1, 784)
            labels = labels.reshape(batch_size, 1)

            model.zero_grad()
            pi, sigma, mu = model(minibatch)
            loss = mdn.mdn_loss(pi, sigma, mu, labels)
            loss.backward()
            optimizer.step()

            print('Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(minibatch), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))

    return model


# get the dataset
train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=batch_size, shuffle=True)

# initialize the model
model = nn.Sequential(
    nn.Linear(784, 400),
    nn.Tanh(),
    nn.Linear(400, 200),
    nn.Tanh(),
    nn.Linear(200, 100),
    nn.Tanh(),
    mdn.MDN(100, 1, 3)
)

# train
num_ep = 3
model = train(model, train_loader, num_ep)

# test
acc = test(model, test_loader)

print("Classification accuracy is ", acc)
