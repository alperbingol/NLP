import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
stopWords = set(stopwords.words('turkish'))
# Start with loading all necessary libraries
import numpy as np
import pandas as pd
from os import path
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

import matplotlib.pyplot as plt
% matplotlib inline
from matplotlib import pylab

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

from nltk.tokenize import sent_tokenize, word_tokenize 
import warnings 

#warnings.filterwarnings(action = 'ignore') 
  
import gensim 
from gensim.models import Word2Vec

import pickle


def create_WordCloud(Docs,dim,wordcloud_outputfile,mode,stopwords):
    
    if(stopwords):
        wc = WordCloud(background_color="white", width=dim*100, height=dim*100,stopwords=stopWords)
    else:
        wc = WordCloud(background_color="white", width=dim*100, height=dim*100)
    # generate word cloud
    print("Please wait, TD-IDF values are calculating...")
    if mode=="TFIDF":
        wc = wc.generate_from_frequencies(term_weight(Docs,mode="TFIDF"))
        
    else:
        wc = wc.generate_from_frequencies(term_weight(Docs))
    
    
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    #plt.show()
    wc.to_file(wordcloud_outputfile)

def term_weight(Docs,mode="TFIDF"):
    #max_df = ignore words appeared in x% of the documents as they are too common
    #binary=True 
    #stopwords=stopWords, priorly eliminate those selected words

    cv = CountVectorizer(Docs,max_df=0.3)
    count_vector=cv.fit_transform(Docs) #tf in doc(in our case one sentence of 4990)
    #print(count_vector.shape)
    #4990 docs, 124754 unique words
    #cv.vocabulary_
    #positions in the sparse vector
    #-----------------------------------------------
    #stopwords selection; prior list, max fr(count), min fr, min idf scores, 
    #to see eliminated words
    #print(cv.stop_words)
    #(count_vector.shape)
    #--------------------------
    #keep counts
    alper = count_vector.toarray()
    df_counts = pd.DataFrame(alper, columns=cv.get_feature_names())
    #df_counts
    el = df_counts.sum(axis=0)
    el_sort = el.sort_values(ascending=False)
    #---------------------------------
    if mode=="TFIDF":
        #IDF values
        tfidf_transformer=TfidfTransformer(smooth_idf=True,use_idf=True) 
        tfidf_transformer.fit(count_vector)
        # print idf values 
        df_idf = pd.DataFrame(tfidf_transformer.idf_, index=cv.get_feature_names(),columns=["idf_weights"]) 
        # sort ascending 
        #df_idf.sort_values(by=['idf_weights'])
        #The lower the IDF value of a word, the less unique it is to any particular document.
        #-----------------------------------------
        #TD-IDF values
        # count matrix, actually same as count_vector but nice to start clean "tf"
        count_vector_tdfidf=cv.transform(Docs) 
        # tf-idf scores 
        tf_idf_vector=tfidf_transformer.transform(count_vector_tdfidf)
        #computing the tf * idf  multiplication where your term frequency is weighted by its IDF values.
        alp = tf_idf_vector.toarray()
        feature_names = cv.get_feature_names() 
        #get tfidf vector for first document 
        #first_document_vector=tf_idf_vector[0]
        #feature_names= unique words 
        #print the scores 
        #df = pd.DataFrame(first_document_vector.T.todense(), index=feature_names, columns=["tfidf"]) 
        df = pd.DataFrame(alp, columns=feature_names) 
        #df.sort_values(by=["tfidf"],ascending=False)
        d = df.sum(axis=0)
        d_sort = d.sort_values(ascending=False)
        return d_sort   
    else:
        return el_sort

def count(Docs):
    cv = CountVectorizer(Docs,stop_words=stopWords,max_df=0.3)
    count_vector=cv.fit_transform(Docs)
    alper = count_vector.toarray()
    df_counts = pd.DataFrame(alper, columns=cv.get_feature_names())
    #df_counts
    el = df_counts.sum(axis=0)
    el_sort = el.sort_values(ascending=False)
    

    return el_sort

def create_HeapsPlot(Docs,heaps_outputfile):
    
    print("Please wait, Tf values are calculating...")
    counts = count(Docs)
    aa=counts.to_frame()
    cumulative_sum = aa.cumsum()
    cumulative_sum.rename(columns = {0:'numbers'}, inplace = True)
    cumulative_sum=cumulative_sum.reset_index()
    cumulative_sum.rename(columns = {"index":'words'}, inplace = True)
    
    #plt.xlabel('term occurence') #size of corpus
    #plt.ylabel('vocabulary size') # number of unique words
    
    cumulative_sum.reset_index().plot(x='index', y='numbers')


def create_ZiphsPlot(Docs,zips_outputfile):
    print("Please wait, Tf values are calculating...")
    freq = count(Docs)
    #aa = freq.to_frame()
    #bb = aa.sort_values(by=0,ascending=False)
    
    n = len(freq)
    ranks = range(1, n+1)                   # x-axis: the ranks
    cc = freq 								#y-axis

    pylab.loglog(ranks, cc) 	#this plots frequency, not relative frequency
    #pylab.loglog(ranks, cc, label='ali')
    pylab.xlabel('log(rank)')
    pylab.ylabel('log(freq)')
    #pylab.legend(loc='lower left')
    pylab.savefig(zips_outputfile)
    pylab.show()


def create_WordVectors(Docs,dim,mode,size):
    
    
    counts = count(Docs)
    aa=counts.to_frame()
    cumulative_sum = aa.cumsum()
    cumulative_sum.rename(columns = {0:'numbers'}, inplace = True)
    cumulative_sum=cumulative_sum.reset_index()
    cumulative_sum.rename(columns = {"index":'words'}, inplace = True)
    unique_words=cumulative_sum['words']
    unique_list = unique_words.tolist()
    
    if mode == "cbow":
        model = gensim.models.Word2Vec(unique_list, min_count = 1, size = dim, window = size)
    
    else:
        model = gensim.models.Word2Vec(unique_list, min_count = 1, size = dim, window = size, sg = 1)
        
    return model

