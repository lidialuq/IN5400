import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import os
import PIL.Image
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
import numpy as np

def get_classes_list():
    classes = ['clear', 'cloudy', 'haze', 'partly_cloudy',
               'agriculture', 'artisinal_mine', 'bare_ground', 'blooming',
               'blow_down', 'conventional_mine', 'cultivation', 'habitation',
               'primary', 'road', 'selective_logging', 'slash_burn', 'water']
    return classes, len(classes)


class ChannelSelect(torch.nn.Module):
    """This class is to be used in transforms.Compose when you want to use selected channels. e.g only RGB.
    It works only for a tensor, not PIL object.
    Args:
        channels (list or int): The channels you want to select from the original image (4-channel).

    Returns: img
    """
    def __init__(self, channels=[0, 1, 2]):
        super().__init__()
        self.channels = channels

    def forward(self, img):
        """
        Args:
            img (Tensor): Image
        Returns:
            Tensor: Selected channels from the image.
        """
        return img[self.channels, ...]

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)


class RainforestDataset(Dataset):
    def __init__(self, root_dir, train, transform):
        self.image_folder = os.path.join(root_dir, 'train-tif-v2')
        self.label_csv = os.path.join(root_dir, 'train_v2.csv')
        self.transform = transform
        
        # Read from pandas and convert to list
        df = pd.read_csv(self.label_csv)
        df['tags'] = df['tags'].str.split()
        label_lst = df['tags'].tolist()
        image_filenames = df['image_name'].tolist()
        # Append all classes to the start of the list so that binarization happens in same order
        classes, _ = get_classes_list()
        label_lst = [classes] + label_lst
        # Binarize
        mlb = MultiLabelBinarizer()
        labels = mlb.fit_transform(label_lst)[1:]

        # Split in train and val
        img_train, img_test, lb_train, lb_test = \
            train_test_split(image_filenames, labels, test_size=0.4, random_state=100)
            
        if train == True: 
            self.img_filenames, self.labels = img_train, lb_train
        elif train == False: 
            self.img_filenames, self.labels = img_test, lb_test
        
            

    def __len__(self):
        return len(self.img_filenames)

    def __getitem__(self, idx):
        img_path = os.path.join( self.image_folder, self.img_filenames[idx]+'.tif')
        img = PIL.Image.open(img_path)
        #to_tensor = transforms.ToTensor()
        #tensor_img = to_tensor(img)
        
        if self.transform is not None: 
            img = self.transform(img)

        sample = {'image': img,
                  'label': self.labels[idx],
                  'filename': self.img_filenames[idx]}
        return sample


if __name__=='__main__':
    trans = transforms.Compose([
            #transforms.ToTensor(),
            ChannelSelect(channels=[0, 1, 2]),
        ]),
    ds = RainforestDataset('/media/lidia/DATA/rainforest/rainforest', train=True, transform=trans)
