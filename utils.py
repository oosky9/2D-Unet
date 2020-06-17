import torch
import torchvision
from torchvision import transforms, datasets
import SimpleITK as sitk
import numpy as np
import os
   
class MyDataset(torch.utils.data.Dataset):

    def __init__(self, data_list, label_list, transform=None):
        self.transform = transform
        self.data = []
        self.label = []
        for d in data_list:
            itk_img = sitk.ReadImage(d)
            img = sitk.GetArrayFromImage(itk_img)
            self.data.append(img.astype(np.float32))
        for l in label_list:
            itk_gt = sitk.ReadImage(l)
            gt = sitk.GetArrayFromImage(itk_gt).astype(bool)
            self.label.append(gt.astype(np.float32))

        assert len(self.data) == len(self.label)
        self.data_num = len(self.data)

    def __len__(self):
        return self.data_num

    def __getitem__(self, idx):
        out_data = self.data[idx]
        out_label = self.label[idx]

        if self.transform:
            out_data = self.transform(out_data)

        return out_data, out_label


def load_dataset(data_path, label_path, batch_size, shuffle):

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([2.9798, ], [14.4141, ])
    ])
    images = MyDataset(data_path, label_path, transform=transform)
    dataloader = torch.utils.data.DataLoader(images, batch_size=batch_size, shuffle=shuffle)

    return dataloader


def debug():

    DATA_DIR = '../train'
    CASE_LIST_PATH = DATA_DIR + '/case_list.txt'

    with open(CASE_LIST_PATH, 'r') as f:
        case_list = [row.strip() for row in f]
    
    d_lis = []
    l_lis = []

    for case in case_list:
        d_lis.append(os.path.join(DATA_DIR, 'Image', case + '.mhd'))
        l_lis.append(os.path.join(DATA_DIR, 'Label', case + '.mhd'))
    
    dl = load_datasets(d_lis, l_lis, 100, True)


if __name__ == '__main__':
    debug()