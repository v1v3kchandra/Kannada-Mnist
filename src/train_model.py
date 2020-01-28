#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 11:40:21 2020

@author: vivek
"""

from keras.models import Sequential
from keras.layers import Conv2D,Flatten,MaxPooling2D,Dense,Dropout
import pandas as pd
import numpy as np
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint

train_csv=pd.read_csv("/train.csv")  # path of training csv file
test_csv=pd.read_csv("/test.csv")    # path of testing csv file
val_csv=pd.read_csv("/val.csv")    # path of validating csv file

x_train=train_csv.iloc[:,1:]    # separating the feature columns from the labels column of training data(features for training)
y_train=train_csv.iloc[:,0]     # separating labels of test dataset from the feature columns(labels for training)
Y_train=np_utils.to_categorical(y_train,10)     # one hot encoding the labels(value 10 because there are 10 classes)

x_val=val_csv.iloc[:,1:]    # features for validating
y_val=val_csv.iloc[:,0]     # labels for validating
Y_val=np_utils.to_categorical(y_val,10)     # one hot encoding the labels(value 10 because there are 10 classes)


img_train=[]   # list to store the numpy value of images for training
for i in range(len(x_train)):
    img_tmp=np.array(x_train.loc[i,:])
    img_train.append(np.reshape(img_tmp,(28,28)))
    img_tmp=[]
    
img_val=[]  # list to store the numpy value of images for testing
for i in range(len(x_val)):
    img_tmp=np.array(x_val.loc[i,:])
    img_val.append(np.reshape(img_tmp,(28,28)))
    img_tmp=[]

images_train=np.array(img_train)
images_train=images_train.reshape(images_train.shape[0], 28, 28, 1)   #keras accepts 4d image during model training i.e, number of images used for training ,image width, image heigt and image depth
img_shape=images_train.shape[1:]  #gives the dimension of the images
images_train=images_train/255   #to map the pixel values from 0-255 to 0-1

images_val=np.array(img_val)
images_val=images_val.reshape(images_val.shape[0], 28, 28, 1)
img_shape_val=images_val.shape[1:]
images_val=images_val/255

################# CNN model #################
model_kannada_mnist=Sequential()
#Block1
model_kannada_mnist.add(Conv2D(30,(5,5),input_shape=img_shape,activation="relu"))
model_kannada_mnist.add(MaxPooling2D(pool_size=(2,2)))#Block2
model_kannada_mnist.add(Conv2D(15,(3,3),activation="relu"))
model_kannada_mnist.add(MaxPooling2D(pool_size=(2,2)))
model_kannada_mnist.add(Dropout(0.5))

#FC
model_kannada_mnist.add(Flatten())
model_kannada_mnist.add(Dense(128,activation="relu"))
model_kannada_mnist.add(Dense(50,activation="relu"))
model_kannada_mnist.add(Dense(10,activation="softmax"))

model_kannada_mnist.compile(loss="categorical_crossentropy",optimizer="adam",metrics=['accuracy'])

print(model_kannada_mnist.summary())
    
model_save_filepath = "/Kan_mnist_model.hdf5"     # mention path where the model needs to be saved

checkpoint=ModelCheckpoint(model_save_filepath,monitor="val_acc",verbose=0,save_best_only=True,save_weights_only=False,mode="auto",period=1)

model_kannada_mnist.fit(images_train,Y_train,epochs=25,verbose=1,validation_data=(images_val,Y_val),callbacks=[checkpoint])


