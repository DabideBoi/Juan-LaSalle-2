import numpy as np
import random
import json

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from nltk_utils import bag_of_words, tokenize, stem
from model import NeuralNet

from nltk.stem import WordNetLemmatizer
lemma = WordNetLemmatizer()

with open('intents.json', 'r') as f:
    intents = json.load(f)

all_words = []
tags = []
xy = []

for intent in intents['intents']:
    tag = intent['tag']
    # add to tag list
    tags.append(tag)
    for pattern in intent['patterns']:
        # tokenize each word in the sentence
        w = tokenize(pattern)
        # add to our words list
        all_words.extend(w)
        # add to xy pair
        xy.append((w, tag))

ignore_words = ['?', '.', '!', "'", "(", ")"]
all_words1 = [lemma.lemmatize(w) for w in all_words if w not in ignore_words]
all_words2 = [stem(w) for w in all_words if w not in ignore_words]

print(all_words1)
print(all_words2)