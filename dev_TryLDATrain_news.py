#!/usr/bin/env python
# coding: utf-8

import sys, os
sys.path.append(os.path.abspath(os.path.join('..', 'module')))
from openTable import *
from filepath import *
from preprocessing import preprocessing_text as pre

# import gensim
from gensim.models.ldamodel import LdaModel
from gensim.models import Phrases,TfidfModel
from gensim import corpora
from gensim.models.coherencemodel import CoherenceModel

from pickle import dump
from json import dumps
from datetime import datetime 
from re import sub
import warnings
warnings.filterwarnings('ignore')

# import spacy
from spacy.lang.id import Indonesian,stop_words
nlp = Indonesian()  # use directly
stopwords = stop_words.STOP_WORDS 
stopwords |= {"nya","jurusan","jurus","the","of"}

def preprocessing(text):
    text = pre.remove_tag(text) #Remove Tag
    text = pre.remove_whitespace(text) #Remove Whitespace
    text = pre.lower(text) #Lower
    text = pre.remove_link(text) #Remove Link
    text = pre.alphabet_only(text) #Get Alphabet
    text = sub(r'sobat pintar','',text) # sorry:(
    text = pre.remove_whitespace(text) #Remove Whitespace
    text = [token.text for token in nlp(text)] #Token
    text = pre.slang(text)
    text = [token.lemma_ for token in nlp(text) if token.lemma_ not in stopwords] #Lemma & stopword
    
    return text

def get_data():
    data,status = open_table(['entryId','content'],'BlogsEntry')
    
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

def tfidf_keyword(corpus,dictionary,mylist):
    model = TfidfModel(corpus=corpus,id2word=dictionary)  # fit model
    model.save('../data/tfidf.h5')
    
    bow = [dictionary.doc2bow(mylist[i]) for i in range(len(mylist))]
    vector = [model[bow[i]] for i in range(len(mylist))]
    vector = [sorted(vector[i], key=lambda tup: tup[1],reverse=True) for i in range(len(vector))] #Sort
    
    feature = []
    result = []
    
    for i in vector:
        for j in range(len(i)):
            feature.append(dictionary[i[j][0]])

        result.append(feature[:150])
        feature = []
        
    return result

def make_corpus(data):
    #Make list of list
    mylist = []
    for i,j in data.iterrows():
        mylist.append(j.content)

    # Add bigrams and trigrams to docs,minimum count 10 means only that appear 10 times or more.
    bigram = Phrases(mylist, min_count=8)
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
    
def save_model(lda_model):
    #Save Model LDA
    lda_model.save('../data/lda.h5')
    
def train():
    dict_encoder = {}
    
    try:
        #Get today Date
        date_end = datetime.today().strftime('%Y-%m-%d')

        #get data
        data = get_data()
        data = rename_column(data,{0:'entryId', 1:'content'})
        data.content = data.content.apply(preprocessing)
        
        #Save to Txt
        with open('../data/train_news.txt', 'a+') as output:
            output.write("Get Today Data Success, {} \n".format(date_end))
        print("Get today Date Success")
    except Exception as e:
        print("Get today Date Failed",e)
        with open('../data/train_news.txt', 'a+') as output:
            output.write("Get Today Data Failed, {} \n".format(date_end))
    
    try:
        #make corpus
        mylist,dictionary,corpus = make_corpus(data)
        
        #make dict encode
        for i,j in zip(range(len(mylist)),data.entryId.tolist()):
            dict_encoder[i] = j
            
        with open('../data/dict_encoder.txt', 'w') as file:
            file.write(dumps(dict_encoder)) # use `json.loads` to do the reverse
        print("Make Dictionary and Corpus Success")

    except Exception as e:
        print("Make Dictionary and Corpus Failed")
        
    try:
        #TF-IDF
        mylist = tfidf_keyword(corpus,dictionary,mylist)
        print("get keyword Success")
    except Exception as e:
        print("get keyword Failed",e)
        
    try:
        start=3
        limit=51
        best_model = get_best_topic(dictionary, corpus=corpus, texts=mylist, start=start, limit=limit)
        
        save_model(best_model)
        print("Train LDA Success")
    except Exception as e:
        print("Train LDA Failed",e)
    
train()