<img src="https://raw.githubusercontent.com/Enzzza/law-scraper/master/media/IEEE.jpg" data-canonical-src="https://raw.githubusercontent.com/Enzzza/law-scraper/master/media/IEEE.jpg" width="600"/>

<br>
<br>

# This code is made for purpose of program IEEE Innovation Nation

## Training NLP model for local languages

<p>With this code we will train our NLP model using local lanugages. To be able to train our model we need to clean text and tokenize it and use lemmatization technique.</p>

```python
import nlzn
import numpy
import tflearn
import tensorflow
import random
import json
import pickle
import os

with open("intents.json",encoding="utf8") as file:
    data = json.load(file)

try:
    with open("data.pickle", "rb") as f:
        words, labels, training, output = pickle.load(f)
except:
    words = []
    labels = []
    docs_x = []
    docs_y = []

    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            wrds = nlzn.word_tokenize(pattern)
            words.extend(wrds)
            docs_x.append(wrds)
            docs_y.append(intent["tag"])

        if intent["tag"] not in labels:
            labels.append(intent["tag"])

    words = sorted(list(set(words)))
    labels = sorted(labels)

    training = []
    output = []

    out_empty = [0 for _ in range(len(labels))]

    for x, doc in enumerate(docs_x):
        bag = []

        for w in words:
            if w in doc:
                bag.append(1)
            else:
                bag.append(0)

        output_row = out_empty[:]
        output_row[labels.index(docs_y[x])] = 1

        training.append(bag)
        output.append(output_row)


    training = numpy.array(training)
    output = numpy.array(output)

    with open("data.pickle", "wb") as f:
        pickle.dump((words, labels, training, output), f)

tensorflow.compat.v1.reset_default_graph()

net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
net = tflearn.regression(net)

model = tflearn.DNN(net)
model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
model.save("model.tflearn")
try:
    model.load("model.tflearn")
except:
    model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
    model.save("model.tflearn")

def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]
    s_words = nlzn.word_tokenize(s)

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1

    return numpy.array(bag)


def chat():
    print("Start talking with the bot!")
    while True:
        inp = input("You:  ")
        if inp.lower() == "quit":
            break

        results = model.predict([bag_of_words(inp,words)])[0]
        results_index = numpy.argmax(results)
        tag = labels[results_index]

        if results[results_index] > 0.7:

            for tg in data["intents"]:
                if tg['tag'] == tag:
                    responses = tg["responses"]

            print(random.choice(responses))
        else:
            print("I didn't get that, try again!")


chat()


```

<p>
Function for tokenization and lemmezation is shown bellow
</p>

```python
"""
@inproceedings{ljubesic-dobrovoljc-2019-neural,
    title = "What does Neural Bring? Analysing Improvements in Morphosyntactic Annotation and Lemmatisation of {S}lovenian, {C}roatian and {S}erbian",
    author = "Ljube{\v{s}}i{\'c}, Nikola  and
      Dobrovoljc, Kaja",
    booktitle = "Proceedings of the 7th Workshop on Balto-Slavic Natural Language Processing",
    month = aug,
    year = "2019",
    address = "Florence, Italy",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/W19-3704",
    doi = "10.18653/v1/W19-3704",
    pages = "29--34"
    }
"""
from bs4 import BeautifulSoup
import string
import re
import classla
from conllu import parse
import ast

classla.download(lang='hr')
nlp = classla.Pipeline('hr', processors='tokenize,lemma')

def word_tokenize(text):

    text = clean_text(text)
    doc = nlp(text)
    tokenlist = parse(doc.to_conll())
    return tokenlist_to_list(tokenlist)

def tokenlist_to_list(tokenlist):
    str_token_list = []
    for i in range(len(tokenlist)):
        for j in range(len(tokenlist[i])):
            token = tokenlist[i][j]
            str_token = repr(token)
            str_token_dict = ast.literal_eval(str_token)
            str_token_list.append(str_token_dict['lemma'])

    return str_token_list

def clean_text(text):
    # remove html tags
    text = BeautifulSoup(text, 'html.parser').get_text()

    # remove special characters and numbers
    pattern = r'[!=\-;:/+,*)@#%(&$_?.^]'
    text =  re.sub(pattern, ' ', text)

    # remove backlash
    text.replace("\\", "")

    # remove extra white spaces and tabs
    pattern = r'^\s*|\s\s*'
    text =  re.sub(pattern, ' ', text).strip()

    return text

```

[Demo](https://www.youtube.com/watch?v=HhkL_ToW4Ns)
