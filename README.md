# Capstone-Project-Ocular-Disease-Classification-Challenge

## Overview

With the integration of Artificial Intelligence (AI) rapidly transforming healthcare, this study explores the use of  Deep Learning techniques for classifying ocular diseases in colour fundus images, leveraging a dataset provided by Peking University. The dataset comprises 5,000 images accompanied by patient data and diagnostic keywords. The objective is to accurately classify diseases into eight categories: normal, diabetes, glaucoma, cataract, AMD, hypertension, myopia, and other abnormalities. Despite challenges with images showing multiple diseases, preprocessing techniques are employed to assign single disease classes effectively.

## Introduction

By 2022/2023, approximately 596 million individuals worldwide were reported to experience  vision impairment, with an additional 510 million facing uncorrected near vision issues. This burden disproportionately affects populations residing in low-income and middle-income countries, highlighting significant disparities in access to eye health services. Moreover, eye health is influenced by conditions that may not immediately impact vision but contribute significantly to the overall unmet need for adequate eye care.

This capstone project serves as a practical application of theoretical knowledge, by focusing on employing Deep Learning techniques, to automatically categorize ocular diseases into 8 Classes using colour fundus images. Deep Learning, a powerful tool for image analysis, is central to this endeavor. Deep Learning models, such as DenseNet121, are very deep with 120 convolution and 4 Avg Pool adept at learning patterns and features from data, which is particularly useful in analyzing medical images like those of the eye. By leveraging Deep Learning techniques, the project aims to streamline and improve disease classification processes, thereby potentially enhancing patient outcomes and healthcare decision-making in the field of ophthalmology.

## Dataset

The ODIR-5K dataset comprises a comprehensive collection of ophthalmic data from 5,000 patients as mentioned above. As shown in the Figure, each patient entry includes age information, color fundus photographs from both left and right eyes, and diagnostic keywords provided by medical professionals. The dataset is categorized into eight distinct classes: normal, diabetes, glaucoma, cataract, age-related macular degeneration (AMD), hypertension, myopia, and other diseases/abnormalities. Classification into these categories was based on the analysis of eye images and patient age. Patient identities were anonymized to safeguard privacy, and the dataset was meticulously divided into training, off-site testing, and on-site testing subsets for evaluation purposes.

<img src="images/capstone.jpg" width="300" alt="The first structured ophthalmic record in ODIR-5K database">
*Figure: The first structured ophthalmic record in ODIR-5K database*

## Preprocessing steps

## 1. Label Mapping:
To address the ambiguity caused by individuals having multiple labels, a preprocessing step is applied to map each individual to a single label.
This simplifies the classification task by providing clarity and consistency.
## 2. Label Categorization:
Labels are grouped into two categories for better distinction:
- Decisive Labels: normal, diabetes, glaucoma, cataract, AMD, hypertension, myopia and Other diseases/abnormalities.
- Non-Decisive Labels:
Lens dust,Optic disk photographically invisible,Low image quality ,Image offset
## 3. Dataset Splitting
The dataset is divided into training and testing subsets:
- Training Set: 90% of the data, used for model training.
- Testing Set: 10% of the data, reserved for evaluating the model's performance on unseen data.
## 4. Data Transformations
To enhance data quality, the following transformations are applied:
- Non-Zero Cropping: Focuses on relevant image regions by removing unnecessary background.
- Padding: Ensures uniform image dimensions.
- Normalization: Standardizes pixel values to improve model performance.
- Resizing: Adjusts images to consistent dimensions for compatibility with the model.

## Model Architecture

The project implemented DenseNet121 model, characterized by dense connectivity within dense blocks, enables each layer to receive feature maps from all preceding layers. These blocks consist of convolutional layers, batch normalization, and ReLU activation functions, promoting efficient feature reuse and propagation. It consists of total 120 convolution and 4 max pool layers. Transition layers inserted between dense blocks reduce feature map dimensions to control computational costs. With a final classification head comprising fully connected layers, DenseNet achieves accurate predictions. In comparison to ResNet-50, DenseNet exhibited superior performance and accuracy in our image classification task. Through experimentation, DenseNet demonstrated higher accuracy, attributed to its parameter efficiency and smooth gradient flow during training. As a result, DenseNet was chosen for its overall better performance, efficiency, and accuracy.

## Training Procedure & Experimentation

The training procedure employed a DenseNet121 model pretrained on ImageNet and fine-tuned specifically for retinal image classification. With a learning rate set to 0.001 and Adam optimizer, the model was trained using mixed precision techniques to enhance efficiency. By freezing all trainable layers except the last classifier layer, the training process was accelerated, enabling quicker convergence. Gradient accumulation was utilized to prevent overfitting, while Ignite provided a streamlined framework for training and evaluation. Moreover, a ReduceLROnPlateau scheduler dynamically adjusted learning rates, optimizing model performance over the course of 10 epochs.

However, the training faced challenges due to an imbalanced dataset, where certain classes were underrepresented compared to others. This imbalance posed obstacles in effectively training the model to accurately classify all classes. Furthermore, the training process was time-consuming, demanding significant computational resources for each epoch. Despite these challenges, the implementation of techniques such as gradient accumulation and mixed precision training helped alleviate some of these issues, ultimately facilitating the development of a robust classification model. 





