# -*- coding: utf-8 -*-
"""Bounding Model.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/15nhNqQAxNTG1-O4zikrNYh5GKMlJBCds
"""

"""from google.colab import drive
drive.mount('/content/drive')
"""

import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Conv2D,MaxPooling2D,Flatten,Activation,Dropout,BatchNormalization
import numpy as np
from tensorflow.keras.optimizers import Adam
from os.path import isfile, join
from os import listdir
import cv2
from PIL import Image
from tensorflow.keras.callbacks import ModelCheckpoint
"""tf.keras.applications.Xception(
    include_top=True,
    weights="imagenet",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=1000,
    classifier_activation="softmax",
)"""
model = Sequential()
inputShape = (128, 128,3)
model.add(Conv2D(32, (3, 3), padding="same",
input_shape=inputShape))
model.add(Activation("relu"))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(64, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(BatchNormalization())
model.add(Conv2D(64, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(128, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(BatchNormalization())
model.add(Conv2D(128, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(BatchNormalization())
model.add(Conv2D(128, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(128, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(BatchNormalization())
model.add(Conv2D(128, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(BatchNormalization())
model.add(Conv2D(128, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(64))
model.add(Activation("relu"))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation("sigmoid"))
adam =Adam()
model.compile( 
  loss='binary_crossentropy',
  optimizer='adam',
  metrics=['accuracy']
)

height=128
width=128
path1='/content/drive/My Drive/bounding box/real/'
realfiles1 = sorted([ f for f in listdir(path1) if isfile(join(path1,f)) ])
#print(realfiles1)
data=[]
labels=[]
###########################################################################
for i in realfiles1:
  try:
    path1='/content/drive/My Drive/bounding box/real/'+i
    img=cv2.imread(path1)
    img=cv2.resize(img,(height,width))
    data.append(img)
    labels.append(0)
  except:
    pass

path2='/content/drive/My Drive/bounding box/fake/'
realfiles2 = sorted([ f for f in listdir(path2) if isfile(join(path2,f)) ])
#print(realfiles2)

for i in realfiles2:
  try:
    path2='/content/drive/My Drive/bounding box/fake/'+i
    img=cv2.imread(path2)
    img=cv2.resize(img,(height,width))
    data.append(img)
    labels.append(1)
  except:
    pass
print(labels)
data=np.array(data)
labels=np.array(labels)
print(len(labels))

checkpoint = ModelCheckpoint(
    '/content/drive/My Drive/bound_model.h5',
    monitor='accuracy',
    save_best_only=True,
    verbose=1
)
model =model.fit( 
  data,
  labels,
  epochs=60,
  callbacks=[checkpoint])