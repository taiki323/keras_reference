from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.callbacks import TensorBoard
from keras.callbacks import EarlyStopping
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.layers.normalization import BatchNormalization
from keras_func import *

train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    'data/train',
    target_size=(150,150),
    batch_size=32,
    class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
    'data/validation',
    target_size=(150,150),
    batch_size=32,
    class_mode='binary')

tensormemory()

model = Sequential()
model.add(Conv2D(32, 3, 3, input_shape=(3, 150, 150),init='he_normal'))
model.add(Activation('relu'))
model.add(MaxPooling2D((2,2)))
model.add(BatchNormalization())

model.add(Conv2D(64,3,3,init='he_normal'))
model.add(Activation('relu'))
model.add(MaxPooling2D((2,2)))
model.add(BatchNormalization())

model.add(Conv2D(64,3,3,init='he_normal'))
model.add(Activation('relu'))
model.add(MaxPooling2D((2,2)))
model.add(BatchNormalization())

model.add(Flatten())
model.add(Dense(64,init='he_normal'))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(BatchNormalization())
model.add(Dense(1,init='he_normal'))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])
model.summary()

tb_cb = TensorBoard(log_dir="/tmp/catdog", histogram_freq=0, write_graph=True)
early_stop = EarlyStopping(patience=50)
hist = model.fit_generator(
    train_generator,
    samples_per_epoch=2000,
    nb_epoch=1,
    validation_data=validation_generator,
    nb_val_samples=800,
    callbacks=[tb_cb])

save_models(model,"teni")
drowplt(hist, "test.jpg")
with open('teni_log.txt','a') as f:
    f.write("'acc:'" + str(hist.history["acc"][-1]) + " 'loss':" + str(hist.history["loss"][-1]) + \
    " 'val_acc'" + str(hist.history["val_acc"][-1]) + " 'val_loss'" + str(hist.history["val_loss"][-1]) + "\n")
