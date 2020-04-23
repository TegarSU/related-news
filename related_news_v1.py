#!/usr/bin/env python
# coding: utf-8

import sys
from os.path import abspath,join
sys.path.append(abspath(join('..', 'module')))
from openTable import *
from AccesDB import *

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

def pivot_table(table):
    pivot = table.pivot_table(index='tagId',columns='classPK',values='count').fillna(0)
    X = pivot.values.T
    
    # fit the model
    SVD = TruncatedSVD(n_components=250, random_state=17)
    matrix = SVD.fit_transform(X)
    # SVD.explained_variance_ratio_.sum()

    corr = np.corrcoef(matrix)
    news_corr = pd.DataFrame(corr, index=pivot.columns, columns=pivot.columns)
    
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
    #Prod
    prod_server,prod_koneksi = Connection()
    
    #DS
    ds_server,ds_koneksi = Connection_2()
    
    entryId = event['entryId']

    today = date.today()
    refreshtime = today - timedelta(days=30)
    statement = ' where entryId = {}'
    recommendation,status = open_table_ds(ds_koneksi,['*'],'related_news_svd',statement=statement.format(entryId))
    
    #First Time
    if not recommendation:
        data = get_data(prod_koneksi)
        news_corr = pivot_table(data)
        result = str(get_recommendation(news_corr,entryId))
#         result = str(result)
        result = literal_eval(result)
        save_recommendation(ds_koneksi,entryId,result,today)
        ds_koneksi.commit()

    else:
        recommendation = recommendation[0]
        recommendation_refreshtime = recommendation[2]
        #Refresh Time
        if refreshtime > recommendation_refreshtime:
            data = get_data(prod_koneksi)
            news_corr = pivot_table(data)
            result = str(get_recommendation(news_corr,entryId))
            result = literal_eval(result)
            statement = ' where entryId = {}'
            data = {
                'recommendation':result,
                'tanggal':today
            }
            status = update_db(ds_koneksi,'related_news_svd',data,statement=statement.format(entryId))
            ds_koneksi.commit()
        #Already Exist
        else:
            result = literal_eval(recommendation[1])
    
    
    ds_koneksi.close()
    ds_server.stop()
    
    prod_koneksi.close()
    prod_server.stop()
    
    return result