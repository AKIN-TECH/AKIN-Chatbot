import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import json 
import numpy as np

from tensorflow import keras
from sklearn.preprocessing import LabelEncoder

import colorama 
colorama.init()
from colorama import Fore, Style, Back

import random
import pickle

with open('AKIN_intents.json', encoding='utf-8-sig') as file:
    data = json.load(file)


def chat():
    model = keras.models.load_model('chat_model')
    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
    with open('label_encoder.pickle', 'rb') as enc:
        lbl_encoder = pickle.load(enc)
    max_len = 250
    while True:
        print(Fore.LIGHTBLUE_EX + "Player: " + Style.RESET_ALL, end="")
        inp = input()
        if inp.endswith("?"):
            inp = inp[:-1] 
        if inp.lower() == "quit":
            break
        result = model.predict(keras.preprocessing.sequence.pad_sequences(tokenizer.texts_to_sequences([inp]),
                                             truncating='post', maxlen=max_len))
        predicted_class = np.argmax(result)
        # confidence = result[0][predicted_class] 
        
        tag = lbl_encoder.inverse_transform([predicted_class])
        for i in data['intents']:
            if i['tag'] == tag:
                print(Fore.GREEN + "AKIN-AI:" + Style.RESET_ALL , np.random.choice(i['responses']))
                # print(Fore.YELLOW + "Confidence:" + Style.RESET_ALL, confidence) 

print(Fore.YELLOW + "Start messaging with the bot (type quit to stop)!" + Style.RESET_ALL)
chat()
