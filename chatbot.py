import random
import json
import numpy as np
import pickle
import os
import datetime
import pywhatkit
import nltk
from nltk.stem import WordNetLemmatizer
import nltk
import tensorflow
from tensorflow import keras
from keras import models
from keras.models import load_model

intents = json.loads(open('intents.json').read())
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('chatbotModel.h5')

lemmatizer = WordNetLemmatizer()

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]), verbose=0)[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
    return return_list

def play_song_on_youtube(song_name):
    pywhatkit.playonyt(song_name)

def search_on_google(search_query):
    pywhatkit.search(search_query)

def get_info(info_query):
    pywhatkit.info(info_query, lines=4)

def get_response(intents_list, intents_json):
    if not intents_list:
        return "Sorry, I didn't understand what you asking for, could you please write again?"

    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']

    for i in list_of_intents:
        if i['tag'] == tag:

            if tag == 'open_app':
                app_name = ' '.join([w for w in clean_up_sentence(message)])
                os.popen(app_name)
                result = random.choice(i['responses']).format(o=app_name)

            elif tag == 'get_date':
                now = datetime.datetime.now()
                result = i['responses'][0] + now.strftime("Date: %d-%m-%Y Time: %H:%M:%S")

            elif tag == 'play_song':
                song_name = ' '.join([w for w in clean_up_sentence(message) if w not in ['play', 'song']])
                play_song_on_youtube(song_name)
                result = i['responses'][0].format(s=song_name)

            elif tag == 'google_search':
                search_query = ' '.join([w for w in clean_up_sentence(message)])
                search_on_google(search_query)
                result = i['responses'][0].format(q=search_query)

            elif tag == 'info_about':
                query = ' '.join([w for w in clean_up_sentence(message)])
                get_info(query)
                result = i['responses'][0].format(i=query)

            else:
                result = random.choice(i['responses'])
            break

    return result

print("Chatbot starting...")

while True:
    message = input("Me: " + "")
    ints = predict_class(message)
    res = get_response(ints, intents)
    print("Bot: " + res)