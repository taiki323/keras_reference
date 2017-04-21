from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.callbacks import TensorBoard
from keras.models import model_from_json
from keras.utils.visualize_util import plot
from keras.callbacks import EarlyStopping
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import matplotlib.pyplot as plt
from keras import backend as K
import tensorflow as tf
from keras.layers.normalization import BatchNormalization


def drowplt(hist):
    plt.subplot(2, 1, 1)
    plt.plot(hist.history['loss'], linewidth=3, label='train')
    plt.plot(hist.history['val_loss'], linewidth=3, label='valid')
    plt.grid()
    plt.legend()
    plt.title("loss")
    plt.xlabel('epoch')
    plt.ylabel('loss')
    # plt.ylim(1e-3, 1e-2)
    #plt.yscale('log')

    plt.subplot(2, 1, 2)
    plt.plot(hist.history['acc'], linewidth=3, label='train')
    plt.plot(hist.history['val_acc'], linewidth=3, label='valid')
    plt.grid()
    plt.legend()
    plt.title("acc")
    plt.xlabel('epoch')
    plt.ylabel('acc')
    # plt.ylim(1e-3, 1e-2)
    #plt.yscale('log')

    plt.tight_layout()
    plt.savefig("plt.png")
    plt.show()


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

config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
K.set_session(session)

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

plot(model, to_file="model_teni.png", show_shapes=True)
json_string = model.to_json()
open('teni.json', 'w').write(json_string)
model.save_weights('teni.h5')

drowplt(hist)
with open('teni_log.txt','a') as f:
    f.write("'acc:'" + str(hist.history["acc"][-1]) + " 'loss':" + str(hist.history["loss"][-1]) + \
    " 'val_acc'" + str(hist.history["val_acc"][-1]) + " 'val_loss'" + str(hist.history["val_loss"][-1]) + "\n")
