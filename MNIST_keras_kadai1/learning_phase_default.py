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
from keras.utils import plot_model

log_dir = "./logs"

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
x_train = preprocess(_x_train)
x_val = preprocess(_x_val)
x_test = preprocess(_x_test)

y_train = preprocess(_y_train, label=True)
y_val = preprocess(_y_val, label=True)
y_test = preprocess(_y_test, label=True)

# 前処理結果確認（実行時はコメントアウト推奨）
# print(x_train.shape)  # (48000, 28, 28, 1)
# print(x_val.shape)  # (12000, 28, 28, 1)
# print(x_test.shape)  # (10000, 28, 28, 1)
# print(x_train.max())  # 1.0
# print(x_val.max())  # 1.0
# print(y_test.max())  # 1.0
# print(y_train.shape)  # (48000, 10)
# print(y_val.shape)  # (12000, 10)
# print(y_test.shape)  # (10000, 10)


# Sequentialモデル: 層を単純に重ねていく1入力1出力モデル
def model_sequential():
    activation = 'relu'

    model = models.Sequential()

    # 参考リンク https://keras.io/ja/layers/convolutional/
    model.add(Conv2D(1, (3, 3), padding='same',
              name='conv1', input_shape=(28, 28, 1)))
    model.add(Activation(activation, name='act1'))
    model.add(MaxPooling2D((2, 2), name='pool1'))

    model.add(Flatten(name='flatten'))
    model.add(Dense(64, name='dense2'))
    model.add(Activation(activation, name='act2'))
    model.add(Dense(10, name='dense3'))
    model.add(Activation('softmax', name='last_act'))

    return model


# Functional API: 多入力多出力モデルに対応したAPI
def model_functional_api():
    activation = 'relu'

    input = Input(shape=(28, 28, 1))

    x = Conv2D(1, (3, 3), padding='same', name='conv1')(input)
    x = Activation(activation, name='act1')(x)
    x = MaxPooling2D((2, 2), name='pool1')(x)

    x = Flatten(name='flatten')(x)
    x = Dense(64, name='dense2')(x)
    x = Activation(activation, name='act2')(x)
    x = Dense(10, name='dense3')(x)
    output = Activation('softmax', name='last_act')(x)

    model = Model(input, output)

    return model



# モデルのコンパイル
model = model_sequential()
model.summary()
plot_model(model, to_file='./logs/model.png',
           show_shapes=True)  # モデルの構成をプリント
model.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])
# optimizer: 最適化を行う関数を決定する(https://keras.io/ja/optimizers/)
# loss: 誤差計算を行う関数を決定する(https://keras.io/ja/losses/)
# metrics: モデルの性能を測る指標を指定する。accuracyは推論した数に対して、何回正解したかを計算する



# コールバック
## 学習中に実行される機能
## 学習中の記録やEarlyStoppingなどの機能を付け加えることができる。
## 詳しくはこちら(https://qiita.com/yukiB/items/f45f0f71bc9739830002)
ckpt_name = 'weights-{epoch:02d}-{loss:.2f}-{accuracy:.2f}-{val_loss:.2f}-{val_accuracy:.2f}.hdf5' # 作成される学習モデルファイルの名前の意味
cbs = [
    TensorBoard(log_dir=log_dir),
    ModelCheckpoint(os.path.join(log_dir, ckpt_name),
                    monitor='val_accuracy', verbose=0,
                    save_best_only=True,  # モデルがmonitorの観点でより良くなった時のみ保存
                    save_weights_only=False,  # 重さのみ保存
                    mode='auto', period=1)
]

# データ拡張について(機能は自分で調べてみてください)
## こちらのサイトでパラメータについて確認(https://qiita.com/takurooo/items/c06365dd43914c253240)
datagen = ImageDataGenerator(
    featurewise_center=False,
    samplewise_center=False,
    featurewise_std_normalization=False,
    samplewise_std_normalization=False,
    zca_whitening=False,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=False,
    vertical_flip=False)

# 学習に必要なパラメータ
batch_size = 128
epochs = 3
verbose = 1
steps_per_epoch = x_train.shape[0] // batch_size

# データ拡張の機能を使いたかったらfit_generator, 学習データのみを使う場合はfitを用いる
## fit_generatorの使い方(https://keras.io/ja/models/model/#:~:text=%E3%81%97%E3%81%9FNumpy%E9%85%8D%E5%88%97%EF%BC%8E-,fit_generator,-fit_generator(generator%2C%20steps_per_epoch)
## fitの使い方(https://keras.io/ja/models/sequential/#:~:text=%27accuracy%27%5D)-,fit,-fit(self%2C%20x)
history = model.fit_generator(
    datagen.flow(x_train, y_train, batch_size=batch_size),
    steps_per_epoch=steps_per_epoch,
    epochs=epochs,
    validation_data=(x_val, y_val),
    callbacks=cbs,
    verbose=verbose)



# 学習経過の確認（学習確認時以外コメントアウト推奨）
## 精度で確認
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

epochs = range(1, len(acc)+1)
plt.plot(epochs, acc, 'b', label='Training accuracy')
plt.plot(epochs, val_acc, 'r', label='Val accuracy')
plt.legend()
plt.savefig("./logs/acc_history.png")
plt.clf()

## 損失で確認
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(loss)+1)
plt.plot(epochs, loss, 'b', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Val loss')
plt.legend()
plt.savefig("./logs/loss_history.png")
plt.clf()


# 最終的な学習したモデルの評価
score = model.evaluate(x_test,  y_test)
print(list(zip(model.metrics_names, score)))

