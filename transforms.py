#transforms
import numpy as np
import  matplotlib as plt
import torch
import torchvision.io as io
from torchvision.transforms import Lambda, Compose,ToTensor,Resize,Normalize
from torchvision.io import read_image
import torchvision.transforms.functional as tvf
from  torchvision.transforms import transforms as T
from PIL import Image

def nonzero_bounding_box(img:np.ndarray, verbose=False):
    '''
    Parameters:
        -img (np.ndarray): Input image in numpy array format.
        -verbose (bool): If True, prints additional details about the image and bounding box.
    Logic:
        1. split the image into four quadrants: h_left_split, h_right_split, w_top_split, w_bottom_split
        2. find the last non-zero pixel position for left and top splits
        3. find the first non-zero pixel position for right and bottom splits
    returns:
         A tuple (left, right, top, bottom), the index of the above 4 values as bounding box (left,top,right,bottom)
    '''
    h,w,c = img.shape


    # split image into four quadrants, use the first channel
    left_half_axis_1d = img[h//2,:w//2,0].tolist()
    top_half_axis_1d = img[:h//2,w//2,0].tolist()

    right_half_axis_1d = img[h//2,w//2:,0].tolist()
    bottom_half_axis_1d = img[h//2:,w//2,0].tolist()

    # find first nonzero pixel positions, if no non-zero pixel positions exist, return lower-bounds and upper-bounds
    try:
        h_left = len(left_half_axis_1d) - left_half_axis_1d[::-1].index(0)
    except ValueError as e:
        # could not find zero in the list
        h_left = 0

    try:
        w_top = len(top_half_axis_1d) - top_half_axis_1d[::-1].index(0)
    except ValueError as e:
        w_top = 0

    try:
        h_right = w//2 + right_half_axis_1d.index(0)
    except ValueError as e:
        h_right = h

    try:
        w_bottom = h//2 + bottom_half_axis_1d.index(0)
    except ValueError as e:
        w_bottom = w

    if verbose:
        print(f'Image size {img.shape}')
        print(h_left,h_right,w_top,w_bottom)
    return h_left,h_right,w_top,w_bottom

def crop_nonzero(img, verbose=False):
    """
    Crops the image to include only the non-zero regions.
    Returns:
        - Cropped image containing only the non-zero regions.

    """
    left, right, top, bottom = nonzero_bounding_box(img,verbose=verbose)
    return img[top:bottom,left:right,:]

def pad_to_largest_square(img:torch.Tensor,verbose=False):
    """
    Parameters:
        - img (torch.Tensor): Input image in tensor format with shape (channels, height, width).
        - verbose (bool): If True, prints details about the padding process.

    Logic:
        1. Determines the largest dimension of the image.
        2. Calculates padding for each side to make the image square.
        3. Applies padding using 'torchvision.transforms.functional.pad'.

    Returns:
        - A padded image tensor of shape (channels, largest_side, largest_side).
        """
    c,h,w = img.shape
    largest_side = max(img.shape)
    if (largest_side - h) != 0 :
        total_pad = largest_side - h
        # this is the side where we need to pad
        if total_pad % 2 == 0:
            #even padding
            top = bottom = total_pad // 2
        else:
            top = total_pad // 2
            bottom = total_pad // 2 + 1
    else:
        top = bottom = 0

    if (largest_side - w )!= 0:
        total_pad = largest_side - w
        # this is the side where we need to pad
        if total_pad % 2 == 0:
            # even padding
            left = right = total_pad // 2
        else:
            # odd padding
            left = total_pad // 2
            right = total_pad // 2 + 1
    else:
        left = right = 0

    required_pad = (left,top,right,bottom)
    padded_img =  tvf.pad(img,required_pad,fill=0,padding_mode='constant')

    if verbose:
        print('Img shape',img.shape)
        print('padding', required_pad)
    return padded_img

def read_image(img_path):
    """
    Loads an image from the specified file path.

    Parameters:
        - img_path (str): Path to the image file.
    Returns:
        - A PIL.Image object of the image in RGB format.
    """
    # img_path = img.filename
    # img = io.read_image(img_path)
    # return img.copy() # return a copy to get rid of UserWarning: The given NumPy array is not writable, and PyTorch does not support non-writable tensors.
    img = Image.open(img_path).convert("RGB")
    return img.copy()
    img_transform = Compose([
    Lambda(read_image),
    Lambda(crop_nonzero),
    ToTensor(),
    Lambda(pad_to_largest_square),
    Normalize(mean= training_img_mean,std= torch.sqrt(training_img_var))
])


def get_img_transform(img_size:int):
    """
    Defines a transformation pipeline for preprocessing images.

    Parameters:
        - img_size (int): Target size to resize the image.

    Returns:
        - A composed transformation that:
        1. Crops the non-zero regions of the image.
        2. Pads it to a square.
        3. Converts it to a tensor.
        4. Normalizes it with training_img_mean and training_img_var.
        5. Resizes it to img_size.
    """
    base_img_transform = img_transform
    resized_img_transform = Compose([
        base_img_transform,
        Resize(size=img_size,interpolation=tvf.InterpolationMode.BILINEAR,antialias=True,)

    ])
    return resized_img_transform

def labels_to_idx(label):
    """
    Converts a label into its corresponding index.

    Parameters:
        - label (str): A label from the LABELS list.

    Returns:
        - The index of the label.
    """
    return LABELS_TO_IDX[label]

label_transform = Lambda(labels_to_idx)



#Predefined mean and variance for normalizing the training images, 
# These values are calculated from the training dataset.
training_img_var = torch.Tensor([0.0713, 0.0345, 0.0140])
training_img_mean = torch.Tensor([0.4384, 0.2866, 0.1646])
#Labels for classification and their corresponding indices
LABELS  = ['N','D','G','C','A','H','M','O']
LABELS_TO_IDX = {l:idx for idx, l in enumerate(LABELS)}