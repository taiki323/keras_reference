# -*- coding: utf-8 -*-
import os
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import optimizers
from keras.callbacks import EarlyStopping
import numpy as np
from collections import Counter

from keras_func import *

img_width, img_height = 150, 150
train_data_dir = 'data/train'
validation_data_dir = 'data/validation'
test_data_dir = 'data/test'
nb_train_samples = 2000
nb_validation_samples = 800
nb_epoch = 50

result_dir = 'results'
if not os.path.exists(result_dir):
    os.mkdir(result_dir)

def save_bottleneck_features():
    """VGG16にDog vs Catの訓練画像、バリデーション画像を入力し、
    ボトルネック特徴量（FC層の直前の出力）をファイルに保存する"""

    # VGG16モデルと学習済み重みをロード
    # Fully-connected層（FC）はいらないのでinclude_top=False）
    model = VGG16(include_top=False, weights='imagenet')
    model.summary()

    # ジェネレータの設定
    datagen = ImageDataGenerator(rescale=1.0 / 255)

    # Dog vs Catのトレーニングセットを生成するジェネレータを作成
    generator = datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=32,
        class_mode=None,
        shuffle=False)

    # ジェネレータから生成される画像を入力し、VGG16の出力をファイルに保存
    bottleneck_features_train = model.predict_generator(generator, 2000)
    np.save(os.path.join(result_dir, 'bottleneck_features_train.npy'),
            bottleneck_features_train)

    # Dog vs Catのバリデーションセットを生成するジェネレータを作成
    generator = datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=32,
        class_mode=None,
        shuffle=False)

    # ジェネレータから生成される画像を入力し、VGG16の出力をファイルに保存
    bottleneck_features_validation = model.predict_generator(generator, 800)
    np.save(os.path.join(result_dir, 'bottleneck_features_validation.npy'),
            bottleneck_features_validation)


def train_top_model():
    """VGGのボトルネック特徴量を入力とし、Dog vs Catの正解を出力とするFCネットワークを訓練"""
    # 訓練データをロード
    # ジェネレータではshuffle=Falseなので最初の1000枚がcats、次の1000枚がdogs
    train_data = np.load(os.path.join(result_dir, 'bottleneck_features_train.npy'))
    train_labels = np.array([0] * int(2000 / 2) + [1] * int(2000 / 2))

    # (2000, 4, 4, 512)
    print(train_data.shape)

    # バリデーションデータをロード
    validation_data = np.load(os.path.join(result_dir, 'bottleneck_features_validation.npy'))
    validation_labels = np.array([0] * int(800 / 2) + [1] * int(800 / 2))

    # (800, 4, 4, 512)
    print(validation_data.shape)
    tensormemory()
    # FCネットワークを構築
    model = Sequential()
    model.add(Flatten(input_shape=train_data.shape[1:]))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    lrate = 0.0001
    for i in range(0,1):
        model.compile(loss='binary_crossentropy',
                      #optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
                      optimizer=optimizers.adam(),
                      #optimizer=optimizers.SGD(lr=0.0002, momentum=0.9),
                      metrics=['accuracy'])
        early_stop = EarlyStopping(patience=20)
        history = model.fit(train_data, train_labels,
                            nb_epoch=50,
                            batch_size=32,
                            validation_data=(validation_data, validation_labels),
                            callbacks=[early_stop]
                            )
        #drowplt(history, "figures/RMSprop_lr=" + str(lrate) + ".png")
        drowplt(history, "figures/teni3.png")
        lrate += 0.0001
    save_history(history, "teni3_result.txt")
    model.save_weights(os.path.join(result_dir, 'bottleneck_fc_model.h5'))

    datagen = ImageDataGenerator(rescale=1.0 / 255)
    generator = datagen.flow_from_directory(
        test_data_dir,
        target_size=(img_width, img_height),
        batch_size=32,
        class_mode=None,
        shuffle=False)
    model1 = VGG16(include_top=False, weights='imagenet')
    bottleneck_features_test = model1.predict_generator(generator, 400)
    print bottleneck_features_test.shape
    labels = np.array([0] * int(400 / 2) + [1] * int(400 / 2))
    result = model.predict_classes(bottleneck_features_test,batch_size=32)
    pred = result.reshape(400)
    count = Counter(pred == labels)
    print "\n" + str(count)
    print "acc:" + str(count[1]/400.0)


if __name__ == '__main__':
    # 訓練データとバリデーションデータのボトルネック特徴量の抽出
    #save_bottleneck_features()

    # ボトルネット特徴量でFCネットワークを学習
    train_top_model()