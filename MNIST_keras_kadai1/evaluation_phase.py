# 参考にしたサイト
# https://qiita.com/takurooo/items/82837e44b466e7634c98
# https://qiita.com/dolce_itf/items/be85244a31654d1d56ef

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

model_name = "./logs/weights-03-0.57-0.83-0.29-0.91.hdf5"  # ./logs/評価したいモデル名.hdf5

# MNISTのデータをロード
## 学習データとテストデータを取得する。
(_x_train_val, _y_train_val), (_x_test, _y_test) = mnist.load_data()
## 学習中の検証データがないので、train_test_split()を使って学習データ8割、検証データを2割に分割する。test_sizeが検証データの割合になっている。
_x_train, _x_val, _y_train, _y_val = train_test_split(
    _x_train_val, _y_train_val, test_size=0.2)

plt.figure(figsize=(10, 10))

# 学習、検証、テストデータの前処理用関数


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


# 前処理を実行
x_test = preprocess(_x_test)

y_test = preprocess(_y_test, label=True)

# 重みのみ学習（save_weights_only）にする場合はこちら
# model = model_sequential()
# model.compile(optimizer='rmsprop',
#               loss='categorical_crossentropy', metrics=['accuracy'])
# model.load_weights(model_name)

model = load_model(model_name)

score = model.evaluate(x_test,  y_test)

# [('loss', 0.03808286426122068), ('acc', 0.9879)]
print(list(zip(model.metrics_names, score)))


for i in range(10):
    data = [(x, t) for x, t in zip(_x_test, _y_test) if t == i]
    x, y = data[0]

    pred = model.predict(preprocess(x, label=False))

    ans = np.argmax(pred)
    score = np.max(pred) * 100

    plt.subplot(5, 2, i+1)
    plt.axis("off")
    plt.title("ans={} score={}\n{}".format(ans, score, ans == y))

    plt.imshow(x, cmap='gray')


plt.tight_layout()
plt.savefig("./logs/prediction_result.png")
plt.show()
