#!/usr/bin/env python
# coding: utf-8

# In[6]:


import sys, os
sys.path.append(os.path.abspath(os.path.join('..', 'module')))
from openTable import *
from filepath import *

from re import sub

import warnings
warnings.filterwarnings('ignore')

# import gensim
from gensim.models.ldamodel import LdaModel
from gensim.models import Phrases
from gensim import corpora

# from ast import literal_eval
from pickle import dump

from gensim.models.coherencemodel import CoherenceModel

import pandas as pd
from json import loads,dumps

# import spacy
from spacy.lang.id import Indonesian,stop_words
nlp = Indonesian()  # use directly
stopwords = stop_words.STOP_WORDS 
stopwords |= {"nya","jurusan","jurus","the","of"}

# #Akronim
def slang(tokenized_sentence):
    slang_word_dict = loads(open("../data/slang_word_dict.txt", 'r').read())

    for index in range(len(tokenized_sentence)):
        for key, value in slang_word_dict.items():
            for v in value:
                if tokenized_sentence[index] == v:
                    tokenized_sentence[index] = key
                else:
                    continue
                    
    return " ".join(tokenized_sentence)

def preprocessing(text):
    text = sub('<[^<]+?>', '', str(text)) #remove tag
    text = text.lower() #lower\n",
    text = sub(r'[^a-z]',' ',str(text)) #get alphabet only
    text = sub(r'\s+', ' ', text) #remove white space
    text = sub(r'sobat pintar','',text) # sorry:(
    text = [token.text for token in nlp(text)] #Token
    text = slang(text)#slang word
    text = sub(r'\s+', ' ', text) #remove white space
    text = [token.lemma_ for token in nlp(text) if token.lemma_ not in stopwords] #Lemma & stopword
    
    return text

def get_data():
    data = open_table(['entryId','content'],'BlogsEntry')
    
    return data

def get_best_topic(dictionary, corpus, texts, limit, start):
    coherence_values = []
    model_list = []
    
    for num_topics in range(start, limit):
        model = LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics, random_state=666)
        model_list.append(model)
        coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
        coherence_values.append(coherencemodel.get_coherence())
        
    #get best model
    max_value = max(coherence_values)
    max_index = coherence_values.index(max_value)
    best_model = model_list[max_index]
        
    return best_model

def make_corpus(data):
    #Make list of list
    mylist = []
    for i,j in data.iterrows():
    #     print(j.content)
    #     tmp = literal_eval(j.content)
        mylist.append(j.content)

    # Add bigrams and trigrams to docs,minimum count 10 means only that appear 10 times or more.
    bigram = Phrases(mylist, min_count=10)
    for idx in range(len(mylist)):
        for token in bigram[mylist[idx]]:
            if '_' in token:
                # Token is a bigram, add to document.
                mylist[idx].append(token)

    # Create Dictionary
    dictionary = corpora.Dictionary(mylist)

    # Term Document Frequency
    corpus = [dictionary.doc2bow(text) for text in mylist]
    
    dump(corpus, open('../data/corpus_LDA.pkl', 'wb'))
    dictionary.save('../data/dictionary_LDA.gensim')
    
    return mylist,dictionary,corpus
    
def save_model(model):
    #Save Model
    model.save('../data/lda.h5')
    
def train():
    dict_encoder = {}

    #get data
    data = get_data()
    data = rename_column(data,{0:'entryId', 1:'content'})
    data.content = data.content.apply(preprocessing)

    #make corpus
    mylist,dictionary,corpus = make_corpus(data)

    #make dict encode
    for i,j in zip(range(len(mylist)),data.entryId.tolist()):
        dict_encoder[i] = j

    with open('../data/dict_encoder.txt', 'w') as file:
         file.write(dumps(dict_encoder)) # use `json.loads` to do the reverse

    start=3
    limit=51
    best_model = get_best_topic(dictionary, corpus=corpus, texts=mylist, start=start, limit=limit)

    save_model(best_model)
    
train()

# t = process_time()
# #do some stuff
# train()
# elapsed_time = process_time() - t
# print(elapsed_time)


# In[1]:


# from time import process_time


# In[1]:


# t = process_time()
# #do some stuff
# train()
# elapsed_time = process_time() - t
# print(elapsed_time)

