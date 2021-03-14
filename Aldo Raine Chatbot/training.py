import random
import json
import pickle
import numpy as np

import nltk 
from nltk.stem import WordNetLemmatizer   #lemmatizer reduces word to its stem to increase script performance

from tensorflow._api.v1.keras import Sequential
from tensorflow._api.v1.keras import Dense, Activation, Dropout
from tensorflow._api.v1.keras.optimizers import SGD


lemmatizer= WordNetLemmatizer

intents=json.loads(open('intents.json').read())  #reads the contents of the json file which gets passed to load function. the json is now a dictionary in


words=[]  #for all of the words we're going to have
classes=[]  #all the classes wer're about to have
documents=[] #combinations/belongings
ignore_letters=['?', '!','.',',']

for intent in intents['intents']:   
    for pattern in intent['patterns']:
        word_list = nltk.word_tokenize(pattern)
        words.append(word_list)
        documents.append((word_list, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

            
print(documents)