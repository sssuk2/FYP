# Used Librarires 
from PIL import Image 
import numpy as np
import pandas as pd
import os
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator, load_img
from keras.layers import Conv2D, Dense, BatchNormalization, Activation, Dropout, MaxPooling2D, Flatten
from keras.optimizers import Adam, RMSprop, SGD
from keras import regularizers
from keras.callbacks import ModelCheckpoint, CSVLogger, TensorBoard, EarlyStopping, ReduceLROnPlateau
import datetime
import matplotlib.pyplot as plt
from keras.utils import plot_model
import matplotlib.image as mpimg
import glob
import cv2
import random
import pickle
 

#Function to Read Images
def load_images_from_folder_test(root_folder,emo_folder):
  all_images = [] 
  all_labels = []
  # Get the classes index and name
  for cls, fol in enumerate(emo_folder): # Read classes of all folders
    print(cls,fol)
    #join the path of train and root folder 
    all_images_per_emo = os.listdir(os.path.join(root_folder, fol)) 
     # Read Images from all sub folder of train
    for im in all_images_per_emo:
      img = cv2.imread(os.path.join(root_folder, fol, im),cv2.IMREAD_GRAYSCALE)
      all_images.append(img)
      all_labels.append(cls)
 
  z = list(zip(all_images, all_labels)) # Zip the images with labels respectively
  random.shuffle(z) # Shuffling the images & labels
  
  all_images, all_labels = zip(*z)  # Unzipping the values after shuffling 
  #print(all_labels[:10])
  return (all_images,all_labels)



#Directories 
root_folder_train = '/content/gdrive/My Drive/Deep Learning FYP/Code/train'
root_folder_test = '/content/gdrive/My Drive/Deep Learning FYP/Code/test/' 
emo_folder_train = os.listdir(root_folder_train) # emo_folder contains 7 folders of training images
emo_folder_test = os.listdir(root_folder_test)   # emo_folder contains 7 folders of testing images

#Loading Traing and Test images into memory
train_images, train_labels = load_images_from_folder_test(root_folder_train,emo_folder_train)
test_images, test_labels = load_images_from_folder_test(root_folder_test,emo_folder_test)

# Saving the Dataset as pickle object
file = open('Dataset on local disk','wb')
obj_1 = ['train_images', train_images]
obj_2 = ['test_images', test_images]
obj_3 = ['train_labels', train_labels]
obj_4 = ['test_labels', test_labels]

pickle.dump(train_images,file)
pickle.dump(train_labels,file)
pickle.dump(test_images,file)
pickle.dump(test_labels,file)
file.close()


