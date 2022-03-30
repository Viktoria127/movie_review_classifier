# -*- coding: utf-8 -*-

import os
import json
import numpy as np
import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential, load_model
from keras.layers import Embedding, Dense, Dropout, GlobalAveragePooling1D
from keras import regularizers
from pathlib import Path

!pip install livelossplot
from livelossplot.tf_keras import PlotLossesCallback

# после выполнения этой клетки вам будет предложено перейти по ссылке, 
# войти в гугл аккаунт и скопировать код в поле ниже. сделайте это и нажмите Enter
from google.colab import drive
drive.mount('/content/drive/')

# считыавние тренировоных данных (текста и рейтингов из json-файла)
texts_neg = []
with open('train_neg.txt', 'r') as f_neg:
    texts_neg = json.load(f_neg)
texts_pos = []
with open('train_pos.txt', 'r') as f_pos:
    texts_pos = json.load(f_pos)    
rankings = []
with open('rankings.txt', 'r') as f_rank:
    rankings = json.load(f_rank)

texts = texts_neg + texts_pos
#test_texts = test_texts_neg + test_texts_pos

# генерация меток отрицательных и положительных отзывов (сначала следуют 12500 отрицательных отзывов, затем 2500 положительных)
labels = [0 for i in range(12500)] + [1 for i in range(12500)]

maxlen = 500 # отсечение длины отзывов после 500-го слова
train_size = 15000 # обучение на выборке из 20000 образцов
valid_size = 10000 # проверка на выборке из 5000 образцов
vocab_size = 10000 # рассмотрение только 10000 наиболее часто используемыx слов

tokenizer = Tokenizer(num_words=vocab_size)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

data = pad_sequences(sequences, maxlen=maxlen)
labels = np.array(labels)
rankings = np.array(rankings)

# Разбивка данных на обучающую и проверочную выборки, но перед этим данные перемешиваются, поскольку отзывы в исходном наборе 
# упорядочуны (сначала следуют отрицательные, а потом положительные)
indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]
rankings = rankings[indices]

x_train = data[:train_size]
y_train = labels[:train_size]
y_train_rank = rankings[:train_size]
x_valid = data[train_size: train_size + valid_size]
y_valid = labels[train_size: train_size + valid_size]
y_valid_rank = rankings[train_size: train_size + valid_size]



path = Path("/content/drive/My Drive/movie_review/pos_neg_best/checkpoint_best.h5")
path.mkdir(exist_ok=True, parents=True) 
assert path.exists()
cpt_filename = "checkpoint_best.h5"  
cpt_path = str(path / cpt_filename)

model = Sequential()
model.add(Embedding(vocab_size, 32, input_length=maxlen))
model.add(GlobalAveragePooling1D())
model.add(Dense(16, activation='relu', kernel_regularizer=regularizers.l2(0.002)))
model.add(Dropout(0.5))
model.add(Dense(16, activation='relu', kernel_regularizer=regularizers.l2(0.002)))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))
model.summary()

# Dense 16, regularizers.l2(0.002), GlobalAveragePooling1D()   # до 15 эпох!!!!!!!!!   /content/drive/MyDrive/movie_review/pos_neg_best
checkpoint = tf.keras.callbacks.ModelCheckpoint(cpt_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

#optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)

model.compile(optimizer='Adam', 
              loss='binary_crossentropy', 
              metrics=["accuracy"]) 
history = model.fit(x_train, y_train,
                    #steps_per_epoch=25,
                    epochs=15,
                    batch_size=32,
                    validation_data=(x_valid, y_valid),
                    callbacks=[PlotLossesCallback(), checkpoint])

# Dense 16, regularizers.l2(0.002), GlobalAveragePooling1D()   # до 15 эпох!!!!!!!!!   /content/drive/MyDrive/movie_review/pos_neg_glob_aver_pool_proba_dense16_l2_0.002
checkpoint = tf.keras.callbacks.ModelCheckpoint(cpt_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

#optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)

model.compile(optimizer='Adam', 
              loss='binary_crossentropy', 
              metrics=["accuracy"]) 
history = model.fit(x_train, y_train,
                    #steps_per_epoch=25,
                    epochs=15,
                    batch_size=32,
                    validation_data=(x_valid, y_valid),
                    callbacks=[PlotLossesCallback(), checkpoint])

# считыавние тестовых данных (текста и рейтингов из json-файла)
test_texts_neg = []
with open('/content/drive/MyDrive/test_neg.txt', 'r') as f_neg:
    test_texts_neg = json.load(f_neg)
test_texts_pos = []
with open('/content/drive/MyDrive/test_pos.txt', 'r') as f_pos:
    test_texts_pos = json.load(f_pos)    
test_rankings = []
with open('/content/drive/MyDrive/rankings_test.txt', 'r') as f_rank:
    test_rankings = json.load(f_rank)

test_texts = test_texts_neg + test_texts_pos

labels_test = [0 for i in range(12500)] + [1 for i in range(12500)]
#tokenizer = Tokenizer(num_words=vocab_size)
#tokenizer.fit_on_texts(test_texts)
sequences = tokenizer.texts_to_sequences(test_texts)
x_test = pad_sequences(sequences, maxlen=maxlen)
y_test = np.asarray(labels_test)
y_test_rank = np.asarray(test_rankings)

restored_model = load_model('/content/drive/My Drive/movie_review/pos_neg_best/checkpoint_best.h5/checkpoint_best.h5') 
loss, acc = restored_model.evaluate(x_test, y_test)

path = Path("/content/drive/My Drive/movie_review/rank_best/checkpoint_best.h5")
path.mkdir(exist_ok=True, parents=True) 
assert path.exists()
cpt_filename = "checkpoint_best.h5"  
cpt_path = str(path / cpt_filename)

cl_weight_dict = {0: 0.6,
 1: 1,
 2: 1,
 3: 1,
 4: 1,
 5: 1,
 6: 1,
 7: 0.6}

model_rank = Sequential()
model_rank.add(Embedding(vocab_size, 32, input_length=maxlen))
model_rank.add(GlobalAveragePooling1D())
model_rank.add(Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.001)))
model_rank.add(Dropout(0.4))
model_rank.add(Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.001)))
model_rank.add(Dropout(0.4))
model_rank.add(Dense(8, activation='softmax'))
model_rank.summary()

# Dense 32, kernel_regularizer=regularizers.l2(0.001), Dropout(0.4) path = Path("/content/drive/My Drive/movie_review/rank_best/checkpoint_best.h5")
checkpoint = tf.keras.callbacks.ModelCheckpoint(cpt_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

model_rank.compile(optimizer=optimizer, 
              loss='sparse_categorical_crossentropy', 
              metrics=['acc']) 
history = model_rank.fit(x_train, y_train_rank,
                         epochs=15,
                         batch_size=32,
                         validation_data=(x_valid, y_valid_rank),
                         class_weight=cl_weight_dict, # используется "упрощенный" словарь весов классов, так как рассчитанный с помощью compute_class_weight
                         # дает слишком большой вес для рейтингов 2 и 9
                         callbacks=[PlotLossesCallback(), checkpoint])

restored_model_rank = load_model('/content/drive/MyDrive/movie_review/rank_best/checkpoint_best.h5/checkpoint_best.h5') 
loss, acc = restored_model_rank.evaluate(x_test, y_test_rank)

def predict_status_ranking():
    #with st.form(key='my_form'):
    text_input = input('Enter a movie review')
    text_list = [0]
    text_list[0] = text_input
    text_input = text_list
    sequences = tokenizer.texts_to_sequences(text_input)
    data = pad_sequences(sequences, maxlen=maxlen)
    predicted_status = restored_model.predict(data)
    predicted_rank = restored_model_rank.predict(data)
    print('Status: ', 'positive' if predicted_status > 0.5 else 'negative')
    print('Ranking: ', (np.argmax(predicted_rank) + 1) if  np.argmax(predicted_rank) <= 3 
          else  (np.argmax(predicted_rank) + 3)) 
        #print(predicted_rank)

predict_status_ranking()

