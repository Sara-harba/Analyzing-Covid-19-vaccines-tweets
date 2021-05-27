#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np



#BOW vectors
def bow_vectors(text_train,text_test,ngram_range,max_features):
    cv = CountVectorizer(ngram_range=ngram_range,max_features=max_features, min_df=0.0, max_df=1.0,binary=False)
    cv_train_features = cv.fit_transform(text_train)
    # transform test articles into features
    cv_test_features = cv.transform(text_test)
    return cv_train_features,cv_test_features



# TF-IDF vectors
def tfidf_vectors(text_train, text_test,ngram_range,max_features):
    tv = TfidfVectorizer(ngram_range=ngram_range,max_features=max_features, min_df=0.0, max_df=1.0,use_idf=True)
    tv_train_features = tv.fit_transform(text_train)
    tv_test_features = tv.transform(text_test)
    return tv_train_features, tv_test_features

