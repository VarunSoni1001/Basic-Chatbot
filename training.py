import random # to get random responses.
import json # to retrieve data from intents.json file.
import numpy as np # numPy is use to make an array of the chatbot trained data.
import pickle # use to make/serialize the training data into .pkl files

import nltk # main component (Used for train and make a chatbotModel.h5 file)
from nltk.stem import WordNetLemmatizer# used to group all similar kind of words into a word


import tensorflow # mainly used for the machine learning of the chatbot, by using model.save('chatbotModel.h5') creates a model for chatbot
from tensorflow import keras
from keras import models, layers, optimizers
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD


lemmatizer = WordNetLemmatizer()
intents = json.loads(open('intents.json').read())

words = []
classes = []
documents = []
ignore_letters = ['?', '!', '.', ',']

for intent in intents['intents']:
    for pattern in intent['patterns']:
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)
        documents.append((word_list, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

words = [lemmatizer.lemmatize(word) for word in words if word not in ignore_letters]
words = sorted(set(words))

classes = sorted(set(classes))
pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

training = []
output_empty = [0] * len(classes)

for document in documents:
    bag = []
    word_patterns = document[0]
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]

    for word in words:
        bag.append(1) if word in word_patterns else bag.append(0)

    output_row = list(output_empty)
    output_row[classes.index(document[1])] = 1
    training.append([bag, output_row])

random.shuffle(training)
training = np.array(training)
# print(len(training))


train_x = list(training[:, 0]) # list of input data
train_y = list(training[:, 1]) # list of corresponding output data


model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

sgd = SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

#epochs ==> dataset iterations (means that the dataset will be passed through the network 'n'(here n=200) times)
#bathc_size ==> training samples used in each iteration (means that the model will be trained on batches of 'n' (here n=5) samples at a time)
# verbose ==> the amount of information printed during training (A value of 'n'(here n=1) means that progress updates will be printed after each epoch of training)

mod = model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1) # .fit fits the model into training data
model.save('chatbotModel.h5')# .h5 means Hierarchical Data Format 5 (HDF5) format, which is a binary file format that is designed to store and organize large amounts of numerical data efficiently

# prints done when completed
print("Done")


# this will show the accuracy percentage when training is completed
loss, accuracy, time = model.evaluate(np.array(train_x), np.array(train_y))
accuracy_percent = accuracy * 100
loss_percentage = loss * 100
print("Test accuracy: {:.2f}%".format(accuracy_percent) + "\n" + "Loss: {:.2f}%".format(loss_percentage))
