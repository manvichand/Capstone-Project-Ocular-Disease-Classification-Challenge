#datasets
from transforms import labels_to_idx,read_image, get_img_transform

# from preprocessing import ROOT_DIR
from torch.utils.data import Dataset
from os.path import join
import csv
# import matplotlib.pyplot as plt
from PIL import Image

# TRAIN_DIR=r"data"
def read_as_csv(csv_path):
    file_name_arr=[]
    label_arr=[]
    with open(csv_path,'r') as f:
        reader=csv.reader(f)
        next(reader)
        for row in reader:
            file_name_arr.append(row[1])
            label_arr.append(row[2])
    return(file_name_arr,label_arr)

class ImageDataset(Dataset):
    def __init__(self,csv_path,transforms:None):
        images,labels=read_as_csv(csv_path)
        self.images = images
        self.labels=labels
        self.transforms=transforms

    def __len__(self):
        return len(self.images)

    def __str__(self):
        return f"<ImageDataset with {self.__len__()} samples>"
    def __getitem__(self,index):
        # image=self.images[index]
        # label=self.labels[index]
        image_name=self.images[index]
        label_name= self.labels[index]

        image_path= join("/content/drive/MyDrive/ODIR-5K_Training_Dataset",image_name)
        # image= Image.open(image_path).convert('RGB')
        label=labels_to_idx(label_name)
        if self.transforms:
            image= self.transforms(image_path)
        return image,label
