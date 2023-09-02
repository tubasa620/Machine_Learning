# 参考にしたサイト
# https://qiita.com/takurooo/items/82837e44b466e7634c98
# https://qiita.com/dolce_itf/items/be85244a31654d1d56ef
# https://www.mikan-tech.net/entry/2020/05/04/201634

# 必要なパッケージのインストール
import os
import re
import keras
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras import models
from keras.models import Model
from keras import Input
from keras.layers import Activation, Conv2D, MaxPooling2D, Flatten, Dense
from keras.callbacks import TensorBoard, ModelCheckpoint
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import load_model
import tensorflow as tf
import numpy as np

# ./logs/評価したいモデル名.hdf5
model_name = "./logs/weights-09-0.10-0.97-0.03-0.99.hdf5"


def load_imgset(filename):
    img = tf.io.decode_image(tf.io.read_file(filename))
    img = tf.image.rgb_to_grayscale(img[:, :, :3])
    img = tf.image.resize(img, [28, 28])
    img = np.reshape(img, (28, 28))
    img / 255.0
    img_set = np.expand_dims(img, 0)  # Create dataset from one image
    return img_set


def preprocess(data, label=False):
    if label:
        # 教師データはto_categorical()でone-hot-encodingする
        data = to_categorical(data)
    else:
        # 入力画像は、astype('float32')で型変換を行い、レンジを0-1にするために255で割る（正規化）
        # 0-255 -> 0-1
        data = data.astype('float32') / 255
        # Kerasの入力データの形式は(ミニバッチサイズ、横幅、縦幅、チャネル数)である必要があるので、reshape()を使って形式を変換する
        # -1は「残りの値から推測される値」という意味となる。reshapeを使うときは1つは-1に設定可能
        # (sample, width, height) -> (sample, width, height, channel)
        data = data.reshape((-1, 28, 28, 1))

    return data

# 重みのみ学習（save_weights_only）にする場合はこちら
# model = model_sequential()
# model.compile(optimizer='rmsprop',
#               loss='categorical_crossentropy', metrics=['accuracy'])
# model.load_weights(model_name)

# モデルの読み込み
model = load_model(model_name)


point = 0
name_list = ["yane","hashimoto","horikoshi","kitamura","kozu","tauchi","yashita"]
for name in name_list:
    print(name)
    for i in range(10):
        img = load_imgset("./free_hand/"+name+"/%d.png" % i)
        pred = model.predict(preprocess(img))
        pred_num = pred.argmax()
        print(i, "-->", pred_num)
        if i == pred_num:
            point += 1
print("Score: ", point, "/", len(name_list)*10)
