"""
https://github.com/harveyslash/Facial-Similarity-with-Siamese-Networks-in-Pytorch/blob/master/Siamese-networks-medium.ipynb
"""

import random
import argparse
import torch
from torch import optim
from torch.utils.data import DataLoader
import torchvision.datasets as datasets

import matplotlib.pyplot as plt

from model import SiameseNetworkDataset, SiameseNetwork, ContrastiveLoss, Config

DEFAULT_IMAGES_DIR = "scraper/images/training/"
DEFAULT_BATCH_SIZE = 64
DEFAULT_NO_EPOCHS = 50
DEFAULT_MODEL_PATH = "model.h5"
DEFAULT_LEARNING_RATE = 0.0005

def show_plot(iteration, loss):
    plt.plot(iteration, loss)
    plt.show()


if __name__ == "__main__":  
    parser = argparse.ArgumentParser(description = "Train T-Shirt Siamese Network")

    parser.add_argument("--images_dir", metavar="images_dir", help="Directory with training images", default=DEFAULT_IMAGES_DIR, type=str)
    parser.add_argument("--batch_size", metavar="batch_size", help="Training batch size", default=DEFAULT_BATCH_SIZE, type=int)
    parser.add_argument("--epochs", metavar="epochs", help="No of epochs to train", default=DEFAULT_NO_EPOCHS, type=int)
    parser.add_argument("--savefile", metavar="modelpath", help="Model File Location", default=DEFAULT_MODEL_PATH, type=str)
    parser.add_argument("--learnrate", metavar="learnrate", help="Learning Rate", default=DEFAULT_LEARNING_RATE, type=float)

    args = parser.parse_args()

    """ Training Config """
    config = Config(args.images_dir, args.batch_size, args.epochs)

    """ Load Dataset """
    images_folder_dataset = datasets.ImageFolder(root=config.images_dir)
    siamese_dataset = SiameseNetworkDataset(imageFolderDataset=images_folder_dataset,
                                        transform=transforms.Compose([transforms.ToTensor()])
                                       ,should_invert=False)

    """ Train Dataset """
    train_dataloader = DataLoader(siamese_dataset,
                        shuffle=True,
                        num_workers=0,
                        batch_size=config.train_batch_size)

    if os.path.exists(args.savefile):
        print("Loading Existing Model")
        net = torch.load(args.savefile)
    else:
        print("Creating New Model")
        net = SiameseNetwork().cuda()
    
    criterion = ContrastiveLoss()
    optimizer = optim.Adam(net.parameters(),lr = args.learnrate )

    
    counter = []
    loss_history = [] 
    iteration_number= 0

    total_step = len(train_dataloader)

    for epoch in range(0, config.train_number_epochs):
        for i, data in enumerate(train_dataloader, 0):
            img0, img1, label = data
            img0, img1, label = img0.cuda(), img1.cuda(), label.cuda()
            optimizer.zero_grad()
            output1, output2 = net(img0, img1)
            loss_contrastive = criterion(output1, output2, label)
            loss_contrastive.backward()
            optimizer.step()
            
            if (i+1) % 10 == 0:
                print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                    .format(epoch+1, config.train_number_epochs, i+1, total_step, loss_contrastive.item()))
                iteration_number += 10
                counter.append(iteration_number)
                loss_history.append(loss_contrastive.item())
        torch.save(net, DEFAULT_MODEL_PATH)
    show_plot(counter, loss_history)





