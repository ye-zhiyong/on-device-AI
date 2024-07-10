import glob
import random
import torch.nn.functional as F
import torchvision
import torch
import os
import torchvision.transforms.functional as f

class SegDataset(torch.utils.data.Dataset):
    def __init__(self, imgs_path):
        self.imgs_path = glob.glob(os.path.join(imgs_path, '*.png'))

    def augment(self, image, label):
        """"Data augmentation, including flip, add contrast's adjustment channel."""

        #Randomly agumentation
        agu_code = random.choice(['H','V','VH','N'])  # 'N' represents original data
        #HorizonFlip
        if agu_code == 'H':
            image = torchvision.transforms.RandomHorizontalFlip(p=1)(image)
            label = torchvision.transforms.RandomHorizontalFlip(p=1)(label)
        #VerticalFlip
        elif agu_code == 'V':
            image = torchvision.transforms.RandomVerticalFlip(p=1)(image)
            label = torchvision.transforms.RandomVerticalFlip(p=1)(label)
        #VerticalFilp + HorizontalFlip
        elif agu_code == 'VH':
            image = torchvision.transforms.RandomVerticalFlip(p=1)(image)
            image = torchvision.transforms.RandomHorizontalFlip(p=1)(image)
            label = torchvision.transforms.RandomVerticalFlip(p=1)(label)
            label = torchvision.transforms.RandomHorizontalFlip(p=1)(label)
        #Rotation 90бу
        #elif agu_code == 'R':
            #image = torchvision.transforms.RandomRotation(degrees=90)(image)
            #label = torchvision.transforms.RandomRotation(degrees=90)(label)
        #Contrast adjustment
        #elif agu_code == 'C':
            #image = torchvision.transforms.ColorJitter(contrast=1.5)(image)
            #label = label
        #Brightness increase
        #elif agu_code == 'BI':
            #image = torchvision.transforms.ColorJitter(brightness=1.5)(image)
            #label = label
        #Brightness decline
        #elif agu_code == 'BD':
            #image = torchvision.transforms.ColorJitter(brightness=0.5)(image)
            #label = label
        #Center crop
        #elif agu_code == 'CC':
            #image = torchvision.transforms.CenterCrop(size=(150, 150))(image)
            #label = torchvision.transforms.CenterCrop(size=(150, 150))(label)
        #Enlarge
        #elif agu_code == 'L':
            #image = torchvision.transforms.Resize(size=(150, 150), antialias=True)(image)
            #label = torchvision.transforms.Resize(size=(150, 150), antialias=True)(label)
        #Ensmall
        #elif agu_code == 'S':
            #image = torchvision.transforms.Resize(size=(600, 600), antialias=True)(image)
            #label = torchvision.transforms.Resize(size=(640, 600), antialias=True)(label)

        return image, label

    def __getitem__(self, index):
        image_path = self.imgs_path[index]
        label_path = image_path.replace('image','label')
        label_path = label_path.replace('jpg', 'png')

        #Read image -> tensor.RGB, [3 * H * W]
        image = torchvision.io.read_image(image_path, mode=torchvision.io.ImageReadMode.RGB)
        #Preprocess label -> tensor, [1 * H * W], 0: background; 1: crack
        label = torchvision.io.read_image(label_path, mode=torchvision.io.ImageReadMode.GRAY)
        label[label==255] = 1
        label[label==0] = 0

        #Data augmentation
        #image, label = self.augment(image,label)

        #Input normanlization
        image = image / 255.0

        return image, label

    def __len__(self):
        return len(self.imgs_path)


