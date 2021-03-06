import sys, os
sys.path.append(os.path.abspath(os.path.join('..', 'module')))
from openTable import *
from AccesDB import *
from preprocessing import preprocessing_text as pre

# import gensim
from gensim.models.ldamodel import LdaModel
from gensim import similarities
from gensim.models import Phrases

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

from datetime import date,timedelta

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
    
    #Load TFIDF
    file = open('../data/tfidf.h5','rb')
    loaded_tfidf = load(file)
    
    #Load encoder
    dict_encoder = loads(open("../data/dict_encoder.txt", 'r').read())
    
    return loaded_model,loaded_corpus,loaded_dict,loaded_tfidf,dict_encoder

def encode(docId,dict_encoder):
    entryId = []
    
    for i in docId:
        result = dict_encoder.get(str(i))
        entryId.append(result)
        
    return entryId    

def new_user(koneksi,event):
    result = []
    entryId = event['entryId']
    try:
        loaded_model,loaded_corpus,loaded_dict,loaded_tfidf,dict_encoder = load_model()
    except Exception as e:
        print(e)
        return result
#         pass
    try:
        #Get Doc
        statement = " WHERE entryId = {}"
        data,status = open_table(koneksi,['entryId','content'],'BlogsEntry',statement=statement.format(entryId))
        text = data[1].values[0]
    except Exception as e:
        print(e)
        return result
#         pass
    try:
        #Test new document
        text = preprocessing(text) #Preprocessing
        # Add bigrams and trigrams to docs,minimum count 10 means only that appear 10 times or more.
        bigram = Phrases([text], min_count=3)
        for token in bigram[text]:
            if '_' in token:
                # Token is a bigram, add to document.
                text.append(token)

        bow = loaded_dict.doc2bow(text)

        vector = loaded_tfidf[bow]  # apply model to the first corpus document
        vector = sorted(vector, key=lambda tup: tup[1],reverse=True) #Sort
        keyword = [loaded_dict[x[0]] for x in vector]

        new_bow = loaded_dict.doc2bow(keyword)
    
        lda_index = similarities.MatrixSimilarity(loaded_model[loaded_corpus])
    
        query = lda_index[loaded_model[new_bow]]
        # # Sort the similarities
        sort_sim = sorted(enumerate(query), key=lambda item: -item[1])
    except Exception as e:
        print(e)
        return result
#         pass
    
    try:
        result = [x[0] for x in sort_sim] #Get Univ ID
        result = encode(result,dict_encoder)
        result = [x for x in result if x !=entryId] #Remove Input EntryId
    except Exception as e:
        print(e)
        return []
#         pass
    
    return result[:10]

def save_recommendation(koneksi,entryId,recommendation,last_update):    
    table = "related_news_lda"
    column = ['entryId','recommendation','tanggal']
    value = [entryId,str(recommendation),last_update]
    status = to_db(koneksi,table,column,value)

def get_similar_article(event):
    #DS
    ds_server,ds_koneksi = Connection_2()
    #Prod
    prod_server,prod_koneksi = Connection()
    
    entryId = event['entryId']

    today = date.today()
    refreshtime = today - timedelta(days=4)
    statement = ' where entryId = {}'
    recommendation,status = open_table_ds(ds_koneksi,['*'],'related_news_lda',statement=statement.format(entryId))

    #First Time
    if not recommendation:
        result = new_user(prod_koneksi,event)
        save_recommendation(ds_koneksi,entryId,result,today)
        ds_koneksi.commit()

    else:
        recommendation = recommendation[0]
        recommendation_refreshtime = recommendation[2]
        #Refresh Time
        if refreshtime > recommendation_refreshtime:
            result = new_user(prod_koneksi,event)
            
            statement = ' where entryId = {}'
            data = {
                'recommendation':result,
                'tanggal':today
            }
            status = update_db(ds_koneksi,'related_news_lda',data,statement=statement.format(entryId))
            ds_koneksi.commit()
        #Already Exist
        else:
            result = literal_eval(recommendation[1])
    
    ds_koneksi.close()
    ds_server.stop()
    
    prod_koneksi.close()
    prod_server.stop()
    
    return result