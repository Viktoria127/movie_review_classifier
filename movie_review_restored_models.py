import json
import tensorflow as tf
import numpy as np
import streamlit as st
from keras.models import load_model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

# считыавние данных, предназначенных для тренировки (текста и рейтингов из json-файла), для обучения токенайзера
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
maxlen = 500    # отсечение длины отзывов после 500-го слова
vocab_size = 10000   # рассмотрение только 10000 наиболее часто используемыx слов
tokenizer = Tokenizer(num_words=vocab_size)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

restored_model_pos_neg = load_model('pos_neg_checkpoint_best.h5')
restored_model_rank = load_model('rank_checkpoint_best.h5')

def predict_status_ranking():
    with st.form(key='my_form'):
        text_input = st.text_input(label='Enter a movie review')
        submit_button = st.form_submit_button(label='Submit')
    text_list = [0]
    text_list[0] = text_input
    text_input = text_list
    sequences = tokenizer.texts_to_sequences(text_input)
    data = pad_sequences(sequences, maxlen=maxlen)
    if submit_button:
        predicted_status = restored_model_pos_neg.predict(data)
        predicted_rank = restored_model_rank.predict(data)
        st.write('Status: ', 'positive' if predicted_status > 0.5 else 'negative')
        st.write('Ranking: ', (np.argmax(predicted_rank) + 1) if  np.argmax(predicted_rank) <= 3 
              else  (np.argmax(predicted_rank) + 3)) 
        #print(predicted_rank)

predict_status_ranking()