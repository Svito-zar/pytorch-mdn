"""
A script for applying MDN to generate MNIST digits
author: Taras Kucherenko
contact: tarask@kth.se
"""

import torch.nn as nn
import torch.optim as optim
import mdn
import torch
from torchvision import datasets, transforms

import matplotlib.pyplot as plt

torch.set_default_tensor_type('torch.FloatTensor')

batch_size = 1000
num_epochs = 10


def plot(ground_truth, produced_images, epoch):
    """
    Plot 3 different pictures: ground truth digits
    alongside with the sample from the model for that digit

    Args:
        ground_truth:     ground truth images from the MNIST dataset
        produced_images:  digits sampled from the MDN
        epoch:            at which epoch are we visualizing

    Returns:
        nothing, saves plots in the 'figures' folder

    """

    fig = plt.figure()
    # Make two raws with three image in each
    ax1 = fig.add_subplot(2, 3, 1)
    ax1.imshow(ground_truth[0][0], cmap='gray')
    plt.title('Original')
    ax2 = fig.add_subplot(2, 3, 2)
    ax2.imshow(ground_truth[1][0], cmap='gray')
    plt.title('Original')
    ax3 = fig.add_subplot(2, 3, 3)
    ax3.imshow(ground_truth[2][0], cmap='gray')
    plt.title('Original')
    ax4 = fig.add_subplot(2, 3, 4)
    ax4.imshow(produced_images[0].reshape((28,28)), cmap='gray')
    plt.title('Sampled')
    ax5 = fig.add_subplot(2, 3, 5)
    ax5.imshow(produced_images[1].reshape((28, 28)), cmap='gray')
    plt.title('Sampled')
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.imshow(produced_images[2].reshape((28, 28)), cmap='gray')
    plt.title('Sampled')

    plt.savefig('figures/results_{}epoch.png'.format(epoch))


if __name__ == "__main__":

    Numb_mix_densities = 8

    # initialize the model
    model = nn.Sequential(
        nn.Linear(10, 100),
        nn.Tanh(),
        nn.Linear(100, 200),
        nn.Tanh(),
        nn.Linear(200, 400),
        nn.Tanh(),
        nn.Linear(400, 800),
        nn.Tanh(),
        mdn.MDN(800, 784, Numb_mix_densities)
    )
    optimizer = optim.Adam(model.parameters(),lr=0.001)

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

    # train the model
    for epoch in range(num_epochs):

        for batch_idx, (labels, minibatch) in enumerate(train_loader):

            gt = labels[0:3]

            minibatch = [[numb==minibatch[i] for numb in range(10)] for i in range(batch_size) ]
            minibatch = torch.FloatTensor(minibatch).unsqueeze(1)

            labels = labels.reshape(batch_size,784)

            model.zero_grad()
            pi, sigma, mu = model(minibatch)
            loss = mdn.mdn_loss(pi, sigma, mu, labels)
            loss.backward()
            optimizer.step()

            print('Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(minibatch), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))

            # Visualize
            if batch_idx == 0:
                samples = mdn.sample(pi, sigma, mu).int()
                images = samples[0:3]
                plot(gt, images, epoch)


    print("The training is compelete! \nVisualization results have been saved in the folder 'figures'")