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
    """
    Parameters:
        -csv_path (str): Path to the CSV file.
    Returns:
        -A tuple containing:
        file_name_arr (list): List of image file names.
        label_arr (list): List of corresponding labels.
    Logic:
        -Opens the CSV file in read mode.
        -Skips the header row.
        -Extracts filenames and labels from subsequent rows.
    """
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
    #A custom dataset class that extends PyTorch's Dataset for handling image data.
    def __init__(self,csv_path,transforms:None):
        images,labels=read_as_csv(csv_path)
        self.images = images
        self.labels=labels
        self.transforms=transforms

    def __len__(self):
        #Returns the total number of samples in the dataset.
        return len(self.images)

    def __str__(self):
        #Provides a string representation of the dataset object.
        return f"<ImageDataset with {self.__len__()} samples>"
    def __getitem__(self,index):
        """
        Retrieves a single sample (image and label) from the dataset.
        Parameters:
            index (int): The index of the sample to retrieve.
        
        Returns:
            A tuple containing:
            image: The preprocessed image tensor (if transformations are applied).
            label: The label index corresponding to the image.
        
        Logic:
            1.Retrieves the image filename and label using the index.
            2.Constructs the full image path.
            3.Converts the label to its corresponding index using labels_to_idx.
            4.Applies transformations (if provided) to the image.
        """
        image_name=self.images[index]
        label_name= self.labels[index]

        image_path= join("/content/drive/MyDrive/ODIR-5K_Training_Dataset",image_name)
        # image= Image.open(image_path).convert('RGB')
        label=labels_to_idx(label_name)
        if self.transforms:
            image= self.transforms(image_path)
        return image,label
