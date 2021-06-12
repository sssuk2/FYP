from PIL import Image 
import numpy as np
import os
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator, load_img
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, ZeroPadding2D, Convolution2D
from keras.layers import Flatten, Dense, Dropout,BatchNormalization, Activation, Lambda
from keras.optimizers import Adam, RMSprop, SGD
from keras.regularizers import l2
from keras.callbacks import ModelCheckpoint, CSVLogger, TensorBoard, EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
from keras.utils import np_utils
from keras import regularizers
import keras.backend as K
from  keras.models import Model,load_model
from keras.layers import Input,Concatenate, concatenate
from keras.applications import inception_v3
import matplotlib.image as mpimg
import pickle
from pathlib import Path
import glob
import cv2
import imageio
from io import BytesIO
import sys
import io
import pickle
import datetime

# Function to find similarity label from images 
def find_sim(name):
  u = []
  sim = 0
  for index,name in enumerate(name):
      # Reading file name 
      f = os.path.basename(name)
      # Finding the similarity location in images
      s = f.find('x')
      # then returing the similarity score based on location of x
      u.append(s)
      # p(u)
      # if x was found in the name return location and skip 
      if u[index] == 1:
        # print(f'sim score is {index+1}')
        sim = (index+1)
        # p(f'sim is {sim}')
        skilfol = True
        break
      # else break the loop 
      else:continue
  return sim

for i,cls in enumerate(os.listdir(os.path.join(root_folder,'train'))):
  print(i,cls)
# os.listdir(os.path.join(root_folder,'train'))

# Function to load triplets from disk 
def load_triplet_images(batch_num,target,dataset_type,which_class):
  # path = '/content/gdrive/My Drive/Deep Learning FYP/Code/'


  path = '/content/gdrive/My Drive/Deep Learning FYP/Code'
  root_folder = os.path.join(path,str(dataset_type))

  one_c = 'ONE_CLASS_TRIPLET'
  two_c = 'TWO_CLASS_TRIPLET/'
  three_c = 'THREE_CLASS_TRIPLET/'  
  trainX = []
  train_1 = []
  train_2 = []
  train_3 = []
  # Similarity scores 
  siml_train = []
  # row_count = 0
  # Loop through classes
  No_of_triplets = (int)(target/3)
  for ind,cls in enumerate(os.listdir(os.path.join(root_folder))):
    if ind != which_class-1: continue
    else:
      print(ind,cls)
      # looping through rows in the classes
      for i,row in enumerate(os.listdir(os.path.join(root_folder,cls))):
        if i == No_of_triplets: break
        else:
          # print(i,row)
          cls_imgs = os.listdir((os.path.join(root_folder,cls,row)))
          # print(row,cls_imgs,find_sim(cls_imgs))
        #extracting the similarity score from the image names
          mode = (find_sim(cls_imgs))
          siml_train.append(mode)
          # # Reading the images from row folders
          img_r = []
          for img in cls_imgs:
            im = cv2.imread(os.path.join(root_folder,cls,row,img))
            res =cv2.resize(im,(224,224))
            img_r.append(res)
          if mode == 1:
            # print('Sim 1')
            train_1.append(img_r[1]) # img2
            train_2.append(img_r[2]) # img3
            train_3.append(img_r[0]) # img1
          elif mode == 2:
            train_1.append(img_r[0]) # img1
            train_2.append(img_r[2]) # img3
            train_3.append(img_r[1]) # img2
          elif mode == 3:
            train_1.append(img_r[0]) # img1
            train_2.append(img_r[1]) # img2
            train_3.append(img_r[2]) # img3
          # print(np.shape(train_1))
          if len(train_1) == batch_num:
            trainX.extend(np.array(train_1))
            trainX.extend(np.array(train_2))
            trainX.extend(np.array(train_3))
            train_1 = []
            train_2 = []
            train_3 = []
        
    
  Xtrain = np.array(trainX)
  Xtrain = Xtrain.reshape(Xtrain.shape[0],224,224,3)
  print(Xtrain.shape)
  Ytrain = np.zeros(shape=(Xtrain.shape[0],224,224,3))
  print(Ytrain.shape)
  print('done')
  return Xtrain,Ytrain,siml_train


# Loading Triplets Two Class triplet
#Total number of triplets in two class  is 705
#  batch_num = 15
#                     load_triplet_images(batch_num,target,dataset_type,which_class)
trainX200,trainY200,siml_train200 = load_triplet_images(15,600,'train',3)
trainX400,trainY400,siml_train400 = load_triplet_images(15,1200,'train',3)
trainX705,trainY705,siml_train705 = load_triplet_images(15,1980,'train',3)

# Load test triplet to test the network performance
testX,testY,siml_test = load_triplet_images(3,45,'test',3)

# ___________________________ FEC Neural Network _________________________________________

DATA_FORMAT='channels_last' # Theano:'channels_first' Tensorflow:'channels_last'
WEIGHT_DECAY=0.0005
LRN2D_NORM=False
USE_BN=True
IM_WIDTH=224
IM_HEIGHT=224
batch_num = 3



#normalization
def conv2D_lrn2d(x,filters,kernel_size,strides=(1,1),padding='same',data_format=DATA_FORMAT,
                 dilation_rate=(1,1),activation='relu',use_bias=True,kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',kernel_regularizer=None,bias_regularizer=None,activity_regularizer=None,
                 kernel_constraint=None,bias_constraint=None,lrn2d_norm=LRN2D_NORM,weight_decay=WEIGHT_DECAY,name=None):
    #l2 normalization
    if weight_decay:
        kernel_regularizer=regularizers.l2(weight_decay)
        bias_regularizer=regularizers.l2(weight_decay)
    else:
        kernel_regularizer=None
        bias_regularizer=None

    x=Conv2D(filters=filters,kernel_size=kernel_size,strides=strides,padding=padding,data_format=data_format,dilation_rate=dilation_rate,
             activation=activation,use_bias=use_bias,kernel_initializer=kernel_initializer,bias_initializer=bias_initializer,
             kernel_regularizer=kernel_regularizer,bias_regularizer=bias_regularizer,activity_regularizer=activity_regularizer,
             kernel_constraint=kernel_constraint,bias_constraint=bias_constraint,name=name)(x)

    if lrn2d_norm:
        #batch normalization
        x=BatchNormalization()(x)

    return x

def inception_module(x,params,concat_axis,padding='same',data_format=DATA_FORMAT,dilation_rate=(1,1),activation='relu',use_bias=True,
                     kernel_initializer='glorot_uniform',bias_initializer='zeros',kernel_regularizer=None,bias_regularizer=None,
                     activity_regularizer=None,kernel_constraint=None,bias_constraint=None,weight_decay=None):
  
    (branch1,branch2,branch3,branch4)=params
    if weight_decay:
        kernel_regularizer=regularizers.l2(weight_decay)
        bias_regularizer=regularizers.l2(weight_decay)
    else:
        kernel_regularizer=None
        bias_regularizer=None
    #1x1
    if branch1[1]>0:
        pathway1=Conv2D(filters=branch1[1],kernel_size=(1,1),strides=branch1[0],padding=padding,data_format=data_format,dilation_rate=dilation_rate,
                        activation=activation,use_bias=use_bias,kernel_initializer=kernel_initializer,bias_initializer=bias_initializer,
                        kernel_regularizer=kernel_regularizer,bias_regularizer=bias_regularizer,activity_regularizer=activity_regularizer,
                        kernel_constraint=kernel_constraint,bias_constraint=bias_constraint)(x)
    #1x1->3x3
    pathway2=Conv2D(filters=branch2[0],kernel_size=(1,1),strides=1,padding=padding,data_format=data_format,dilation_rate=dilation_rate,
                    activation=activation,use_bias=use_bias,kernel_initializer=kernel_initializer,bias_initializer=bias_initializer,
                    kernel_regularizer=kernel_regularizer,bias_regularizer=bias_regularizer,activity_regularizer=activity_regularizer,
                    kernel_constraint=kernel_constraint,bias_constraint=bias_constraint)(x)

    pathway2=Conv2D(filters=branch2[1],kernel_size=(3,3),strides=branch1[0],padding=padding,data_format=data_format,dilation_rate=dilation_rate,
                    activation=activation,use_bias=use_bias,kernel_initializer=kernel_initializer,bias_initializer=bias_initializer,
                    kernel_regularizer=kernel_regularizer,bias_regularizer=bias_regularizer,activity_regularizer=activity_regularizer,
                    kernel_constraint=kernel_constraint,bias_constraint=bias_constraint)(pathway2)

    #1x1->5x5
    pathway3=Conv2D(filters=branch3[0],kernel_size=(1,1),strides=1,padding=padding,data_format=data_format,dilation_rate=dilation_rate,
                    activation=activation,use_bias=use_bias,kernel_initializer=kernel_initializer,bias_initializer=bias_initializer,
                    kernel_regularizer=kernel_regularizer,bias_regularizer=bias_regularizer,activity_regularizer=activity_regularizer,
                    kernel_constraint=kernel_constraint,bias_constraint=bias_constraint)(x)

    pathway3=Conv2D(filters=branch3[1],kernel_size=(5,5),strides=branch1[0],padding=padding,data_format=data_format,dilation_rate=dilation_rate,
                    activation=activation,use_bias=use_bias,kernel_initializer=kernel_initializer,bias_initializer=bias_initializer,
                    kernel_regularizer=kernel_regularizer,bias_regularizer=bias_regularizer,activity_regularizer=activity_regularizer,
                    kernel_constraint=kernel_constraint,bias_constraint=bias_constraint)(pathway3)

    #3x3->1x1
    pathway4=MaxPooling2D(pool_size=(3,3),strides=branch1[0],padding=padding,data_format=DATA_FORMAT)(x)
    if branch4[0]>0:
        pathway4=Conv2D(filters=branch4[0],kernel_size=(1,1),strides=1,padding=padding,data_format=data_format,dilation_rate=dilation_rate,
                        activation=activation,use_bias=use_bias,kernel_initializer=kernel_initializer,bias_initializer=bias_initializer,
                        kernel_regularizer=kernel_regularizer,bias_regularizer=bias_regularizer,activity_regularizer=activity_regularizer,
                        kernel_constraint=kernel_constraint,bias_constraint=bias_constraint)(pathway4)
    if branch1[1]>0:
        return concatenate([pathway1,pathway2,pathway3,pathway4],axis=concat_axis)
    else:
        return concatenate([pathway2, pathway3, pathway4], axis=concat_axis)

def conv_block(input, nb_filter, dropout_rate=None, weight_decay=1E-4):
    x = Activation('relu')(input)
    x = Convolution2D(nb_filter, (3, 3), kernel_initializer="he_uniform", padding="same", use_bias=False,
                      kernel_regularizer=l2(weight_decay))(x)
    if dropout_rate is not None:
        x = Dropout(dropout_rate)(x)
    return x


def dense_block(x, nb_layers, nb_filter, growth_rate, dropout_rate=None, weight_decay=1E-4):
    concat_axis = 1 if K.image_data_format() == "th" else -1
    feature_list = [x]

    for i in range(nb_layers):
        x = conv_block(x, growth_rate, dropout_rate, weight_decay)
        feature_list.append(x)
        x = Concatenate(axis=concat_axis)(feature_list)
        nb_filter += growth_rate

    return x, nb_filter

def l2_norm(x):
    x = x ** 2
    x = K.sum(x, axis=1)
    x = K.sqrt(x)
    return x


def triplet_loss(y_true, y_pred):
    batch = batch_num
    # anchor
    ref1 = y_pred[0:batch,:]
    # positive
    pos1 = y_pred[batch:batch+batch,:]
    # negative
    neg1 = y_pred[batch+batch:3*batch,:]

    # acnhor - poss
    dis_pos = K.sum(K.square(ref1 - pos1), axis=1, keepdims=True)
    # anchor - negative
    dis_neg = K.sum(K.square(ref1 - neg1), axis=1, keepdims=True)
    # positive - negative
    dis_pneg = K.sum(K.square(pos1 - neg1), axis=1, keepdims=True)
    a1pha = 0.2
    
    d1 = K.maximum(0.0,(dis_pos-dis_neg)+a1pha)
    d2 = K.maximum(0.0,(dis_pos-dis_pneg)+a1pha) 
    d = d1+d2
    return K.mean(d)


def create_model(img_input):
    #Data format:tensorflow,channels_last;theano,channels_last
    if DATA_FORMAT=='channels_first':
        INP_SHAPE=(3,224,224)
        img_input=Input(shape=INP_SHAPE)
        CONCAT_AXIS=1
    elif DATA_FORMAT=='channels_last':
        INP_SHAPE=(224,224,3)
        img_input=Input(shape=INP_SHAPE)
        CONCAT_AXIS=3
    else:
        raise Exception('Invalid Dim Ordering')

    x=conv2D_lrn2d(img_input,64,(7,7),2,padding='same',lrn2d_norm=False,name="FaceNet_NN2_conv2D")
    x=MaxPooling2D(pool_size=(3,3),strides=2,padding='same',data_format=DATA_FORMAT)(x)
    x=BatchNormalization()(x)

    x=conv2D_lrn2d(x,64,(1,1),1,padding='same',lrn2d_norm=False)

    x=conv2D_lrn2d(x,192,(3,3),1,padding='same',lrn2d_norm=True)
    x=MaxPooling2D(pool_size=(3,3),strides=2,padding='same',data_format=DATA_FORMAT)(x)

    x=inception_module(x,params=[(1,64),(96,128),(16,32),(32,)],concat_axis=CONCAT_AXIS) #3a
    x=inception_module(x,params=[(1,64),(96,128),(32,64),(64,)],concat_axis=CONCAT_AXIS) #3b
    #x=MaxPooling2D(pool_size=(3,3),strides=2,padding='same',data_format=DATA_FORMAT)(x)
    x = inception_module(x, params=[(2,0), (128, 256), (32, 64), (0,)], concat_axis=CONCAT_AXIS)  # 3c

    x=inception_module(x,params=[(1,256),(96,192),(32,42),(128,)],concat_axis=CONCAT_AXIS) #4a
    x=inception_module(x,params=[(1,224),(112,224),(32,64),(128,)],concat_axis=CONCAT_AXIS) #4b
    x=inception_module(x,params=[(1,192),(128,256),(32,64),(128,)],concat_axis=CONCAT_AXIS) #4c
    x=inception_module(x,params=[(1,160),(144,288),(32,64),(128,)],concat_axis=CONCAT_AXIS) #4d
    x=inception_module(x,params=[(2,0),(160,256),(64,128),(0,)],concat_axis=CONCAT_AXIS) #4e
    #x=MaxPooling2D(pool_size=(1,1),strides=1,padding='same',data_format=DATA_FORMAT,name="EndOfNN2")(x)

    x = Convolution2D(512, (1, 1), kernel_initializer="he_uniform", padding="same", name="DenseNet_initial_conv2D", use_bias=False,
                      kernel_regularizer=l2(WEIGHT_DECAY))(x)

    x = BatchNormalization()(x)

    x, nb_filter = dense_block(x, 5, 512, growth_rate=64,dropout_rate=0.5)

    x = AveragePooling2D(pool_size=(7, 7), strides=1, padding='valid', data_format=DATA_FORMAT)(x)

    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(16)(x)
    x = Lambda(lambda x: K.l2_normalize(x))(x)

    return x, img_input
# ______________________________ Callbacks and Model compilation _________________________

reduce_learning_rate = ReduceLROnPlateau(monitor='loss',factor=0.5,patience=2,min_lr=0.00001,verbose=1)
callbacks = [reduce_learning_rate]
K.clear_session()
x, img_input = create_model()
model = Model(inputs=img_input,outputs=[x])
# model.summary()
model.compile(loss=triplet_loss, optimizer=Adam(learning_rate=0.0005))

# _________________________ Model Training ________
K.clear_session()
H705 = model.fit(x = trainX705,y = trainY705,batch_size= 3*batch_num,epochs = 100,shuffle=True,callbacks= callbacks)

# Ploting training Results 
# summarize history for loss
plt.figure(figsize=(18,8)) 
# plt.subplot(121)
plt.plot(H400.history['loss'])
plt.title('model loss 400 Triplets')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['400 Triplets loss'], loc='upper right')
plt.savefig('400_triplets')
           
# Save model to avoid to save time
  model2.save('training/705_triplets')
  
# Generating emeddigns from model to evaluate how close wether results are right as comapred to similarty scores
result = model3.predict(testX3)

# Function to compare the embedding vector of images 
def comparison(image_triplet):
  array = image_triplet
  r = array[0]
  p = array[1]
  n = array[2]
  d0  = K.sum(K.square(r-p))
  d1  = K.sum(K.square(r-n))
  d2  = K.sum(K.square(p-n))
  
  
  return (d0.numpy(),d1.numpy(),d2.numpy())
 
  
# Compare embeddings
comp = comparison(result[3:6])
print(comp)
print('Similarity label is: ',siml_test[1])
plt.figure()
plt.subplot(131)
plt.imshow(testX3[3])
plt.subplot(132)
plt.imshow(testX3[4])
plt.subplot(133)
plt.imshow(testX3[5])
print('Img1 distance > img2',comp[0]>comp[1])
print('Img1 distance > img3',comp[0]>comp[2])
print('Img2 distance > img3',comp[1]>comp[2])        


# _________________ End of file _______________
