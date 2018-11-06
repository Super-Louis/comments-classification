# -*- coding: utf-8 -*-
# Author  : Super~Super
# FileName: model.py
# Python  : python3.6
# Time    : 18-8-9 23:59

import json
import numpy as np
from sklearn.model_selection import train_test_split
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint

# load word2num dict
with open('../data/word2num', 'r') as f:
    word2num = json.load(f)

X = np.load('../data/X_lstm.npy')[:10000]
Y = np.load('../data/Y_lstm.npy')[:10000]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)

# 循环神经网络阶段长度
max_len = 100
batch_size = 100

# 长度补全或截断
# todo: change the max len or use None
X_train = sequence.pad_sequences(X_train, maxlen=max_len, value=word2num['padded'])
X_test = sequence.pad_sequences(X_test, maxlen=max_len, value=word2num['padded'])

# 构建模型
model = Sequential()
model.add(Embedding(len(word2num),128))
# 只会得到最后一个节点的输出，如果要得到每个节点的输出，可将return_sequence设置为True
model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
# 构建最后的全连接层, 最后输出维度为1
model.add(Dense(1, activation='sigmoid'))
cp = ModelCheckpoint('./models/model.{epoch:02d}-{val_loss:.2f}.hdf5', monitor='val_loss', verbose=0,
                         save_best_only=False, save_weights_only=False, mode='auto', period=1)

# compile
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
model.fit(X_train, Y_train, batch_size=batch_size, epochs=10, validation_data=(X_test, Y_test), callbacks=[cp])

# evaluate
score = model.evaluate(X_test, Y_test, batch_size=batch_size)
print('Test loss:', score[0])
print('Test accuracy', score[1])
model.save('lstm_keras2')

