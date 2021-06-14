from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset
import torch

class CelebDS(Dataset):
    def __init__(self, imgtransform, anntransform):
        self.imgtransform = imgtransform
        self.anntransform = anntransform
        self.ds = ImageFolder('celeba/img_align_celeba', transform = imgtransform)
        
    def __getitem__(self, idx):
        return (self.ds[idx][0], self.anntransform(idx))

    def __len__(self):
        return len(self.ds)

def celeb_label_transform(desired_attr):
    """
    Pass in a the names of all the attributes that you want
    """

    file = open('celeba/list_attr_celeba.csv').read().split()
    attr_names = file[0].split(',')
    file = file[1:]
    
    def transform(idx):
        attr = torch.tensor([int(entry) for entry in file[idx].split(',')[1:]])
        mask = [attr_names[1:][i] in desired_attr for i in range(len(attr))]
        masked = attr[mask]
        return torch.relu(masked).float()
    return transform

def dummy_label_transform():
    def transform(idx):
        return torch.tensor([0, 0])
    return transform

file = open('celeba/list_attr_celeba.csv').read().split()
attr_names = file[0].split(',')
