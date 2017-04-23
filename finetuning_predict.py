# -*- coding: utf-8 -*-
import os
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model
from keras.layers import Activation, Dropout, Flatten, Dense, Input
from keras import optimizers
from keras.callbacks import EarlyStopping
import numpy as np
from collections import Counter
from keras.models import model_from_json

input_tensor = Input(shape=(150, 150, 3))
vgg16_model = VGG16(include_top=False, weights='imagenet', input_tensor=input_tensor)

top_model = Sequential()
top_model.add(Flatten(input_shape=vgg16_model.output_shape[1:]))
top_model.add(Dense(256, activation='relu'))
top_model.add(Dropout(0.5))
top_model.add(Dense(1, activation='sigmoid'))

# VGG16とFCを接続
model = Model(input=vgg16_model.input, output=top_model(vgg16_model.output))
model.load_weights("finetuning.h5")

model.compile(loss='binary_crossentropy',
                  optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
                  metrics=['accuracy'])
model.summary()

test_datagen = ImageDataGenerator(rescale=1.0 / 255)

test_generator = test_datagen.flow_from_directory(
    "data/test",
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary',
    shuffle=False)

labels = np.array([0] * int(400 / 2) + [1] * int(400 / 2))
result = model.predict_generator(test_generator,400)
pred = result.reshape(400)
pred[pred >= 0.5] = [1]
pred[pred < 0.5] = [0]
count = Counter(pred == labels)
print "\n" + str(count)
print "acc:" + str(count[1]/400.0)