#!/usr/bin/env python
# coding: utf-8

import sys
from os.path import abspath,join
sys.path.append(abspath(join('..', 'module')))
from openTable import *
from AccesDB import *

from scipy.sparse import csr_matrix
from pandas.api.types import CategoricalDtype

import sklearn
from sklearn.decomposition import TruncatedSVD
import numpy as np
from datetime import date,timedelta
from ast import literal_eval

def get_data(koneksi):
    data_blog,status = open_table(koneksi,['entryId'],'BlogsEntry')
    data_blog = data_blog.rename(columns={0: 'classPK'})
    
    statement = " WHERE classNameId = 20011"
    assetentry,status = open_table(koneksi,['entryId','classPK'],'AssetEntry',statement = statement)
    assetentry = assetentry.rename(columns={0:'entryId', 1:'classPK'})
    entryTag,status = open_table(koneksi,['entryId','tagId'],'AssetEntries_AssetTags')
    entryTag = entryTag.rename(columns={0:'entryId', 1:'tagId'})

    data = merge_table(data_blog,assetentry,key='classPK',how='right')
    data = merge_table(data,entryTag,key='entryId')
    
    #Select column
    data['count'] = 1
    
    return data

def pivot(data):
    data['count'] = 1

    person_c = CategoricalDtype(sorted(data.classPK.unique()), ordered=True)
    thing_c = CategoricalDtype(sorted(data.tagId.unique()), ordered=True)

    row = data.classPK.astype(person_c).cat.codes
    col = data.tagId.astype(thing_c).cat.codes
    sparse_matrix = csr_matrix((data["count"], (row, col)),shape=(person_c.categories.size, thing_c.categories.size))
    data = pd.DataFrame.sparse.from_spmatrix(sparse_matrix,index=person_c.categories, columns=thing_c.categories)
    
    return data

def svd_model(table):
    pivot_data = pivot(table)
    X = pivot_data.values

    # fit the model
    SVD = TruncatedSVD(n_components=250, random_state=17)
    matrix = SVD.fit_transform(X)
    # SVD.explained_variance_ratio_.sum()

    corr = np.corrcoef(matrix)
    news_corr = pd.DataFrame(corr, index=pivot_data.index, columns=pivot_data.index)
    
    return news_corr



def get_recommendation(news_corr,entryId):
    recommendation = list(news_corr[entryId].sort_values(ascending=False).head(11).index.values)
    recommendation.remove(entryId)
    
    return recommendation[:10]

def save_recommendation(koneksi,entryId,recommendation,date):   
    table = "related_news_svd"
    column = ['entryId','recommendation','tanggal']
    value = [entryId,str(recommendation),date]
    status = to_db(koneksi,table,column,value)
    
def related_news_1(event):
    related = []
        
    entryId = event['entryId']
    #check recommendation
    try:
        #DS
        ds_server,ds_koneksi = Connection_2()
#         print('open connection 2')
        today = date.today()
        refreshtime = today - timedelta(days=30)
        statement = ' where entryId = {}'
        recommendation,status = open_table_ds(ds_koneksi,['*'],'related_news_svd',statement=statement.format(entryId))
        ds_koneksi.close()
        ds_server.stop()
#         print('close connection 2')
#         print('check recommendation')
    except:
        ds_koneksi.close()
        ds_server.stop()
#         print('close connection')        
    
    #First Time
    if not recommendation:
#         print('first time')
        #get data
        try:
            #Prod
            prod_server,prod_koneksi = Connection()
#             print('open connection')
            
            data = get_data(prod_koneksi)
            
            prod_koneksi.close()
            prod_server.stop()
#             print('close connection')
#             print('open table')
        except:
            prod_koneksi.close()
            prod_server.stop()
#             print('close connection')
            
            return related
        #svd
        try:
            news_corr = svd_model(data)
#             print('svd')
        except:            
            return related
        #get recommendation
        try:           
            result = str(get_recommendation(news_corr,entryId))
            result = literal_eval(result)
#             print('get recommendation')
        except Exception as e:   
            print(e)
            return related
        
        #save db
        try:
            ds_server,ds_koneksi = Connection_2()
            
            save_recommendation(ds_koneksi,entryId,str(result),today)
            ds_koneksi.commit()
#             print('save')
            
            ds_koneksi.close()
            ds_server.stop()
#             print('close connection 2')
        except:
            ds_koneksi.close()
            ds_server.stop()
#             print('close connection 2')
            
            return related

    else:
        recommendation = recommendation[0]
        recommendation_refreshtime = recommendation[2]
#         print('get refresh time')
        #Refresh Time
        if refreshtime > recommendation_refreshtime:
            #get data
            try:
                #Prod
                prod_server,prod_koneksi = Connection()
#                 print('open connection')

                data = get_data(prod_koneksi)

                prod_koneksi.close()
                prod_server.stop()
#                 print('close connection')
#                 print('open table')
            except:
                prod_koneksi.close()
                prod_server.stop()
#                 print('close connection')

                return related
            #svd
            try:
                news_corr = svd_model(data)
#                 print('svd')
            except:            
                return related
            #get recommendation
            try:           
                result = str(get_recommendation(news_corr,entryId))
#                 result = literal_eval(result)
#                 print('get recommendation')
            except:           
                return related          

            statement = ' where entryId = {}'
            data = {
                'recommendation':result,
                'tanggal':today
            }
            try:
                ds_server,ds_koneksi = Connection_2()
                status = update_db(ds_koneksi,'related_news_svd',data,statement=statement.format(entryId))
                ds_koneksi.commit()
#                 print('save')

                ds_koneksi.close()
                ds_server.stop()
#                 print('close connection 2')
            except:
                ds_koneksi.close()
                ds_server.stop()
#                 print('close connection 2')
            
        #Already Exist
        else:
            result = literal_eval(recommendation[1])
#             print('already exist')
    
    return result