import pandas as pd
import os
import numpy as np
from preprocessing import get_individual_labels, test_normal,test_others
from glob import glob
import matplotlib.pyplot as plt


from pandas import DataFrame
from pathlib import Path
# from torchvision.transforms import img_transform,crop_nonzero
from sklearn.model_selection import train_test_split
# from transforms.transforms import
# import sys
# sys.path.append("../")
# import cv2


TRAIN_DIR="/content/drive/MyDrive"
df=pd.read_excel('/content/drive/MyDrive/ODIR-5K_Training_Annotations(Updated)_V2.xlsx')
csv_data= df.to_csv(os.path.join(TRAIN_DIR,"data.csv"))

Left_eye=df[['Left-Fundus','Left-Diagnostic Keywords']].copy()
Left_eye.columns=['Image','Labels']
# Left_eye.to_csv(os.path.join(TRAIN_DIR,'left_eye.csv'))

Right_eye=df[['Right-Fundus','Right-Diagnostic Keywords']].copy()
Right_eye.columns=['Image','Labels']
# Right_eye.to_csv(os.path.join(TRAIN_DIR,'right_eye.csv'))


keywords_left=[keyword_l for keywords_left in df['Left-Diagnostic Keywords'] for keyword_l in keywords_left.split(',')]
unique_keywords_left= set(keywords_left)
# print((unique_keywords_left))
# print(keywords_left[:10])

keywords_right=[keyword_r for keywords_right in df['Right-Diagnostic Keywords'] for keyword_r in keywords_right.split(',')]
unique_keywords_right= set(keywords_right)
# print((unique_keywords_right))
# print(keywords_right[:10])

class_labels= ['N','D','C','A','F','M','O']
keyword_label_mapping  = {
    'normal':'N',
    'retinopathy':'D',
    'glaucoma':'G',
    'cataract':'C',
    'macular degeneration':'A',
    'hypertensive':'H',
    'myopia':'M',
    'lens dust':'O', 'optic disk photographically invisible':'O', 'low image quality':'O', 'image offset':'O'
}
non_decisive_labels = ["lens dust", "optic disk photographically invisible", "low image quality", "image offset"]
# print(get_individual_labels('optic disk photographically invisible'))

df['Left-label']= df['Left-Diagnostic Keywords'].apply(get_individual_labels)
df['Right-label'] = df['Right-Diagnostic Keywords'].apply(get_individual_labels)

df[df['Left-label'].isin(non_decisive_labels)]
df[df['Right-label'].isin(non_decisive_labels)]

#for lefteye.csv

# left_data= pd.read_csv(r'data\left_eye.csv')
# left_columns = 'left_labels'
# l=[]
# for left in left_data['Labels']:
#      out_l= get_individual_labels(left)
#      l.append(out_l)


# left_data[left_columns]=l
# # print(l)
# left_data.to_csv(r'C:\Users\Dell\Desktop\grandchallenge\data\left_eye.csv',index=False)

#for righteye.csv

# right_data= pd.read_csv(r'data\right_eye.csv')
# right_columns = 'right_labels'
# r=[]
# for right in right_data['Labels']:
#      out_r= get_individual_labels(right)
#      r.append(out_r)


# right_data[right_columns]=r
# print(r)
# right_data.to_csv(r'C:\Users\Dell\Desktop\grandchallenge\data\right_eye.csv',index=False)



# find rows where both left and right have beeen processed as Normal, but the final diagnosis is not 'N
df[df.apply(test_normal, axis=1) == False]
# find rows where none of the left and right have been processed as Others, but the final diagnosis also contains 'O'
df[df.apply(test_others,axis=1) == False]

#transforms

# img_paths = glob(f'{df}/*.jpg')
# # len(img_paths)
# for i in range(len(img_paths)//10):
#     fig,ax = plt.subplots(1,2)
#     img = plt.imread(img_paths[i*10])
#     cropped_img = crop_nonzero(img)
#     ax[0].imshow(img)
#     ax[1].imshow(cropped_img)
#     plt.show()



    # create a new dataframe where each row corresponds to one image
left_fundus = df['Left-Fundus']
left_label = df['Left-label']
left_keywords = df['Left-Diagnostic Keywords']
right_fundus = df['Right-Fundus']
right_label = df['Right-label']
right_keywords = df['Right-Diagnostic Keywords']
id = df['ID']
age = df['Patient Age']
sex = df['Patient Sex']

# separate train and test split

SEED = 234
id_train, id_val = train_test_split(id,test_size=0.1,random_state=SEED)

train_left_fundus = df[df['ID'].isin(id_train)]['Left-Fundus']
train_left_label = df[df['ID'].isin(id_train)]['Left-label']
train_left_keywords = df[df['ID'].isin(id_train)]['Left-Diagnostic Keywords']

train_right_fundus = df[df['ID'].isin(id_train)]['Right-Fundus']
train_right_label = df[df['ID'].isin(id_train)]['Right-label']
train_right_keywords = df[df['ID'].isin(id_train)]['Right-Diagnostic Keywords']


val_left_fundus = df[df['ID'].isin(id_val)]['Left-Fundus']
val_left_label = df[df['ID'].isin(id_val)]['Left-label']
val_left_keywords = df[df['ID'].isin(id_val)]['Left-Diagnostic Keywords']

val_right_fundus = df[df['ID'].isin(id_val)]['Right-Fundus']
val_right_label = df[df['ID'].isin(id_val)]['Right-label']
val_right_keywords = df[df['ID'].isin(id_val)]['Right-Diagnostic Keywords']

# stack left and right columns vertically
train_fundus = pd.concat([train_left_fundus, train_right_fundus],axis=0,ignore_index=True,sort=True)
train_label = pd.concat([train_left_label,  train_right_label],axis=0,ignore_index=True,sort=True)
train_keywords = pd.concat([train_left_keywords,train_right_keywords],axis=0,ignore_index=True,sort=True)

val_fundus = pd.concat([val_left_fundus, val_right_fundus],axis=0,ignore_index=True)
val_label = pd.concat([val_left_label,val_right_label],axis=0,ignore_index=True)
val_keywords = pd.concat([val_left_keywords,val_right_keywords],axis=0,ignore_index=True)

train_df_left_right_separate_row = pd.concat([train_fundus,
                                              train_label,
                                              train_keywords],axis=1,sort=True,
                                              keys = ['fundus','label','keywords']) # stack horizontally
val_df_left_right_separate_row = pd.concat([  val_fundus,
                                              val_label,
                                              val_keywords],axis=1,sort=True,
                                              keys=['fundus','label','keywords']) # stack horizontally

cleaned_train_df = train_df_left_right_separate_row.drop(train_df_left_right_separate_row[train_df_left_right_separate_row['label'].isin(non_decisive_labels)].index)
cleaned_val_df = val_df_left_right_separate_row.drop(val_df_left_right_separate_row[val_df_left_right_separate_row['label'].isin(non_decisive_labels)].index)
cleaned_train_df.to_csv('/content/drive/MyDrive/processed_train_.csv')
cleaned_val_df.to_csv('/content/drive/MyDrive/processed_val-5K.csv')
# len(df),len(id_train),len(id_val),len(train_fundus),len(val_fundus),len(train_df_left_right_separate_row),len(val_df_left_ri



# test_root_dir = '../odir2019/ODIR-5K_Testing_Images'
# test_paths = glob(f'{test_root_dir}/*.jpg')
# test_paths = [ Path(p).name for p in test_paths]
# test_df = DataFrame(data={'fundus':test_paths})
# test_df.to_csv('../csv/processed_test_ODIR-5k.csv')








