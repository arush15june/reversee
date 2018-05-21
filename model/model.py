import random
import numpy as np
import pandas as pd
from PIL import Image
import PIL.ImageOps    
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.utils
import torchvision.transforms as transforms

class Config():

    def __init__(self, images_dir, batch_size, epochs, *args, **kwargs):
        self.images_dir = images_dir
        self.train_batch_size = batch_size
        self.train_number_epochs = epochs

class QueryModel():
    """
        Get top 10 similar products from the database by passing a PIL.Image
    """
    def __init__(self, model_path, dataset, *args, **kwargs):
        self.model_path = model_path
        self.net = torch.load(model_path).cuda()
        self.dataloader = DataLoader(dataset, num_workers=0, batch_size=1, shuffle=False)
        self.data_df = pd.DataFrame(columns=['image', 'dissimilarilty'])
        self.data_list = []

    @staticmethod
    def resizeImage(image):
        return image.resize((130, 150), Image.ANTIALIAS)
    
    def ImageToTensor(self, image):
        image = self.resizeImage(image)
        transform=transforms.Compose([transforms.ToTensor()])
        return torch.reshape(transform(image), [1, 3, 150, 130])

    def getDissimilarity(self, tensor0, tensor1):
        output0, output1 = self.net(Variable(tensor0).cuda(), Variable(tensor1).cuda())
        euclidean_distance = F.pairwise_distance(output0, output1)
        return euclidean_distance.item()

    def predict(self, image):
        """ image: PIL Image """
        self.data_list = []
        dataiter = iter(self.dataloader)
        query_image = self.ImageToTensor(image)

        self.data_list = [
            {
                'image': image_name[0],
                'dissimilarity': self.getDissimilarity(query_image, image)
            } 
            for image_name, image in dataiter
            ]

        self.data_df = pd.DataFrame(self.data_list)
        self.data_df = self.data_df.sort_values('dissimilarity').reset_index(drop=True)
        return self.data_df[:10]


class QueryDataset():
    def __init__(self, imageFolderDataset, transform=None):
        self.imageFolderDataset = imageFolderDataset
        self.transform = transform
    
    @staticmethod
    def getImage(file_name):
        return Image.open(file_name)
    
    @staticmethod
    def resizeImage(image):
        return image.resize((130, 150), Image.ANTIALIAS)

    def __getitem__(self, index):
        image_file_name = self.imageFolderDataset.imgs[index][0]
        image = self.getImage(image_file_name)
        image = self.resizeImage(image)

        if self.transform is not None:
            image = self.transform(image)

        return image_file_name, image 
    
    def __len__(self):
        return len(self.imageFolderDataset.imgs)
            
class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.cnn1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(3, 4, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(4),
            
            nn.ReflectionPad2d(1),
            nn.Conv2d(4, 8, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(8),


            nn.ReflectionPad2d(1),
            nn.Conv2d(8, 8, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(8),


        )

        self.fc1 = nn.Sequential(
            nn.Linear(8*130*150, 500),
            nn.ReLU(inplace=True),

            nn.Linear(500, 500),
            nn.ReLU(inplace=True),

            nn.Linear(500, 5))

    def forward_once(self, x):
        output = self.cnn1(x)
        output = output.view(output.size()[0], -1)
        output = self.fc1(output)
        return output

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2

class SiameseNetworkDataset(Dataset):
    """ 
    Get the dataset for the network, the aim is to get image pairs randomly where 50% 
    image pairs belong to the same class and 50% to a different class
    """
    def __init__(self, imageFolderDataset, transform=None, should_invert=True):
        self.imageFolderDataset = imageFolderDataset    
        self.transform = transform
        self.should_invert = should_invert
        
    def getRandomImage(self):
        return random.choice(self.imageFolderDataset.imgs)

    def getRandomLabel(self):
        return np.where(np.random.random() > 0.5, 1, 0)
    
    @staticmethod
    def getImage(file_name):
        return Image.open(file_name)#.convert("L")

    @staticmethod
    def generateLabel(label):
        return torch.from_numpy(np.array([int(label)], dtype=np.float32))

    def __getitem__(self, index):
        img0_tuple = self.getRandomImage()

        """ we need to make sure approx 50% of images are in the same class """
        should_get_same_class = self.getRandomLabel()

        if should_get_same_class:
            img1_tuple = img0_tuple
        else:
            img1_tuple = self.getRandomImage()

        """ open images and convert to to grayscale """
        img0 = self.getImage(img0_tuple[0])
        img1 = self.getImage(img1_tuple[0])
        
        if self.should_invert:
            img0 = PIL.ImageOps.invert(img0)
            img1 = PIL.ImageOps.invert(img1)

        if self.transform is not None:
            img0 = self.transform(img0)
            img1 = self.transform(img1)
        
        return img0, img1 , self.generateLabel(should_get_same_class)

    def __len__(self):
        return len(self.imageFolderDataset.imgs)

class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2)
        loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))


        return loss_contrastive
