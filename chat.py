import random
import json
import nltk
import torch

from model import NeuralNet
from nltk_utils import bag_of_words, tokenize

device = torch.device('cpu')

nltk.download('all')

with open('intents.json', 'r') as json_data:
    intents = json.load(json_data)

FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "Juan La Salle"

cache = {}

def get_response(msg):
    msg = msg.lower()
    print("Loading Cache")
    if msg in cache:
        print("Input located, returning response")
        return cache[msg]
    else:
        print("Input not found, processing response")
        sentence = tokenize(msg)
        X = bag_of_words(sentence, all_words)
        X = X.reshape(1, X.shape[0])
        X = torch.from_numpy(X).to(device)
        output = model(X)
        #print("Output: ",output)
        _, predicted = torch.max(output, dim=1)
        #print("prediction: ", predicted)
        tag = tags[predicted.item()]
        #print("tag: ", tags[predicted.item()])
        probs = torch.softmax(output, dim=1)
        #print("softmax: ", probs)
        prob = probs[0][predicted.item()]
        #print("probability: ", prob.item())
        if prob.item() > 0.85:
            for intent in intents['intents']:
                if tag == intent["tag"]:
                    resp = random.choice(intent['responses'])
                    cache[msg] = resp
                    print(cache)
                    return resp
        
        return "Sorry, my data is currently insufficient to properly handle your inquiry. You may contact the Registrar's Office through phone or email.\n\nEmail: registrar@dlsud.edu.ph\nPhone: 555-5555"


if __name__ == "__main__":
    print("Let's chat! (type 'quit' to exit)")
    while True:
        # sentence = "do you use credit cards?"
        sentence = input("You: ")
        if sentence == "quit":
            break
        
        resp = get_response(sentence)
        print(resp)

