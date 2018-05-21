import os
from PIL import Image
import torchvision.datasets as datasets
from torchvision import transforms
from model import QueryModel, QueryDataset

class ModelPredictor():
    def __init__(self, model_path, images_dir):
        self.image_dataset = datasets.ImageFolder(root=images_dir)
        self.queryDataset = QueryDataset(
                                    imageFolderDataset=self.image_dataset,
                                    transform=transforms.Compose([transforms.ToTensor()]),
                                    )

        self.model = QueryModel(model_path, self.queryDataset)

    @staticmethod
    def getImage(image_file):
        image_file.seek(0)
        image = Image.open(image_file)
        return image

    def _predict(self, image):
        """ image: PIL Image """
        return self.model.predict(image)

    def getMatches(self, image_file):
        """ image_file: File Name/BytesIO Object """
        image = self.getImage(image_file)

        top10 = self._predict(image)

        pathToResource = lambda path: os.path.basename(path)

        top10['image'] = top10['image'].apply(pathToResource)

        return top10

