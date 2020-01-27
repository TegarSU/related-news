import sys, os
sys.path.append(os.path.abspath(os.path.join('..', 'module')))
from openTable import *

import warnings
warnings.filterwarnings('ignore')

# import gensim
from gensim.models.ldamodel import LdaModel

from re import sub

# import spacy
from spacy.lang.id import Indonesian,stop_words
nlp = Indonesian()  # use directly
stopwords = stop_words.STOP_WORDS 
stopwords |= {"nya","jurusan","jurus","the","of"}

from json import loads
from ast import literal_eval
from pickle import load

from gensim import similarities

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

def load_model():
    #Load Model
    loaded_model = LdaModel.load('../data/lda.h5')
    
    #Load Corpus
    file = open('../data/corpus_LDA.pkl','rb')
    loaded_corpus = load(file)
    
    #Load Dictionary
    file = open('../data/dictionary_LDA.gensim','rb')
    loaded_dict = load(file)
    
    #Load encoder
    dict_encoder = loads(open("../data/dict_encoder.txt", 'r').read())
    
    return loaded_model,loaded_corpus,loaded_dict,dict_encoder

def encode(docId,dict_encoder):
    entryId = []
    
    for i in docId:
        result = dict_encoder.get(str(i))
        entryId.append(result)
        
    return entryId    

def get_similar(event):
    try:
        entryId = event['entryId']
        loaded_model,loaded_corpus,loaded_dict,dict_encoder = load_model()
    except:
        pass
    
    try:
        #Get Doc
        statement = " WHERE entryId = {}"
        data = open_table(['entryId','content'],'BlogsEntry',statement=statement.format(entryId))
        text = data[1].values[0]
    except:
        pass
    
    try:
        #Test new document
        bow = loaded_dict.doc2bow(preprocessing(text))
    
        lda_index = similarities.MatrixSimilarity(loaded_model[loaded_corpus])
    
        query = lda_index[loaded_model[bow]]
        # # Sort the similarities
        sort_sim = sorted(enumerate(query), key=lambda item: -item[1])
    except:
        pass
    
    try:
        result = [x[0] for x in sort_sim] #Get Univ ID
        result = encode(result,dict_encoder)
        result.remove(entryId) #Remove Input EntryId
    except:
        pass
    
    return result[:5]