import os
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.models import Sequential
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import optimizers
import numpy as np
from keras.applications.vgg16 import VGG16, preprocess_input,decode_predictions
from keras_func import *


filename = "/home/ubtaiki/Picture/nougi/G0174231.JPG"

tensormemory()
model = VGG16(include_top=True, weights='imagenet')
model.summary()

img = image.load_img(filename, target_size=(224,224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)

preds = model.predict(preprocess_input(x))
results = decode_predictions(preds, top=5)[0]
for result in results:
    print(result)