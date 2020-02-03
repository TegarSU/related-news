import sys, os
sys.path.append(os.path.abspath(os.path.join('..', 'module')))
from openTable import *
from preprocessing import preprocessing_text as pre

# import gensim
from gensim.models.ldamodel import LdaModel
from gensim import similarities

# import spacy
from spacy.lang.id import Indonesian,stop_words
nlp = Indonesian()  # use directly
stopwords = stop_words.STOP_WORDS 
stopwords |= {"nya","jurusan","jurus","the","of"}

from json import loads
from ast import literal_eval
from pickle import load
from re import sub
import warnings
warnings.filterwarnings('ignore')

def preprocessing(text):
    text = pre.remove_tag(text) #Remove Tag
    text = pre.lower(text) #Lower
    text = pre.remove_link(text) #Remove Link
    text = pre.alphabet_only(text) #Get Alphabet
    text = sub(r'sobat pintar','',text) # sorry:(
    text = pre.remove_whitespace(text) #Remove Whitespace
    text = [token.text for token in nlp(text)] #Token
    text = pre.slang(text)
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
    except Exception as e:
#         print(e)
        pass
    
    try:
        #Get Doc
        statement = " WHERE entryId = {}"
        data = open_table(['entryId','content'],'BlogsEntry',statement=statement.format(entryId))
        text = data[1].values[0]
    except Exception as e:
#         print(e)
        pass
    
    try:
        #Test new document
        bow = loaded_dict.doc2bow(preprocessing(text))
    
        lda_index = similarities.MatrixSimilarity(loaded_model[loaded_corpus])
    
        query = lda_index[loaded_model[bow]]
        # # Sort the similarities
        sort_sim = sorted(enumerate(query), key=lambda item: -item[1])
    except Exception as e:
#         print(e)
        pass
    
    try:
        result = [x[0] for x in sort_sim] #Get Univ ID
        result = encode(result,dict_encoder)
        result.remove(entryId) #Remove Input EntryId
    except Exception as e:
#         print(e)
        pass
    
    return result[:5]