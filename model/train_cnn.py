"""
https://github.com/harveyslash/Facial-Similarity-with-Siamese-Networks-in-Pytorch/blob/master/Siamese-networks-medium.ipynb
"""

import os
import argparse
import pandas as pd
import torch
from torch import optim
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
from torchvision import transforms

import matplotlib.pyplot as plt

from model import ConvNet, ConvNetDataset, Config

DEFAULT_IMAGES_DIR = "scraper/images/training/"
DEFAULT_BATCH_SIZE = 200
DEFAULT_NO_EPOCHS = 50
DEFAULT_MODEL_PATH = "model_cnn.h5"
DEFAULT_LEARNING_RATE = 0.0005
DFEAULT_LABEL_FILE = "classes.csv"

def show_plot(iteration, loss):
    plt.plot(iteration, loss)
    plt.show()


if __name__ == "__main__":  
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    parser = argparse.ArgumentParser(description = "Train T-Shirt Siamese Network")

    parser.add_argument("--images_dir", metavar="images_dir", help="Directory with training images", default=DEFAULT_IMAGES_DIR, type=str)
    parser.add_argument("--batch_size", metavar="batch_size", help="Training batch size", default=DEFAULT_BATCH_SIZE, type=int)
    parser.add_argument("--epochs", metavar="epochs", help="No of epochs to train", default=DEFAULT_NO_EPOCHS, type=int)
    parser.add_argument("--savefile", metavar="modelpath", help="Model File Location", default=DEFAULT_MODEL_PATH, type=str)
    parser.add_argument("--learnrate", metavar="learnrate", help="Learning Rate", default=DEFAULT_LEARNING_RATE, type=float)
    parser.add_argument("--labelfile", metavar="labelfile", help="Classes File", default=DFEAULT_LABEL_FILE, type=str)

    args = parser.parse_args()
    print(args)

    """ Training Config """
    config = Config(args.images_dir, args.batch_size, args.epochs)

    """ Load Dataset """
    images_folder_dataset = datasets.ImageFolder(root=config.images_dir)
    cnn_dataset = ConvNetDataset(imageFolderDataset=images_folder_dataset,
                                        classes='classes.csv',
                                        transform=transforms.Compose([transforms.ToTensor()])
                                        )
    labels = pd.read_csv(args.labelfile)

    """ Train Dataset """
    train_dataloader = DataLoader(cnn_dataset,
                        shuffle=True,
                        num_workers=0,
                        batch_size=config.train_batch_size)
                        
    if os.path.exists(args.savefile):
        print("Loading Existing Model")
        net = torch.load(args.savefile)
    else:
        print("Creating New Model")
        net = ConvNet(num_classes=len(labels)).cuda()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr = args.learnrate)

    counter = []
    loss_history = [] 
    iteration_number= 0

    total_step = len(train_dataloader)
    for epoch in range(config.train_number_epochs):
        for i, (image_file_names, images, labels) in enumerate(train_dataloader):
            images = images.cuda()
            labels = labels.cuda()
            
            # Forward pass
            outputs = net(images)
            loss = criterion(outputs, labels)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if (i+1) % 10 == 0:
                print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                    .format(epoch+1, config.train_number_epochs, i+1, total_step, loss.item()))
                iteration_number += 10
                counter.append(iteration_number)
                loss_history.append(loss.item())
        torch.save(net, args.savefile)
    show_plot(counter, loss_history)





