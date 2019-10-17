# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 07:31:12 2019

@author: TREAF ALSHEMERI
@email: treafalshamari@gmail.com
"""


import numpy as np
import matplotlib.pyplot as plt 
from keras.models import Sequential
from keras.layers import Conv2D , Flatten , Dense , Dropout , Activation , MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator

train_dir = '../x-ray/train/' #Path of Train Dataset
test_dir = '../x-ray/test/' #Path of Test Dataset
val_dir = '../x-ray/val/' #Path of Validation Dataset

batch_size = 32
input_shape = (150,150,3) #(Width , Height , RGB)

#Architecturing The Model 
model = Sequential()
model.add(Conv2D(32,(3,3),input_shape = input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = (2,2)))
model.add(Conv2D(32,(3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = (2,2)))
model.add(Conv2D(64 , (3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = (2,2)))
model.add(Conv2D(64, (3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = (2,2)))
model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.25))
model.add(Dense(1))
model.add(Activation('sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
#Training Augmentation
train_datagen = ImageDataGenerator( rescale=1. / 255,shear_range=0.2,zoom_range=0.2,horizontal_flip=True)
#Testing Augmentation
test_datagen = ImageDataGenerator(rescale=1. / 255)
train_generator = train_datagen.flow_from_directory(train_dir,target_size=(150, 150),batch_size=batch_size,class_mode='binary')
validation_generator  = test_datagen.flow_from_directory(val_dir,target_size=(150, 150),batch_size=batch_size,class_mode='binary')
test_generator = test_datagen.flow_from_directory(test_dir,target_size=(150, 150),batch_size=batch_size,class_mode='binary')
model.fit_generator( train_generator,steps_per_epoch=326,epochs=20, validation_data=validation_generator,validation_steps=16)
#Saving the model
model.save('CNN_model.h5')
#Evaluating the model
scores = model.evaluate_generator(test_generator)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
