#!/usr/bin/env python
# coding: utf-8

# In[2]:


import qalsadi.lemmatizer
from stop_words import get_stop_words
from nltk.tokenize import word_tokenize
from nltk.stem.isri import ISRIStemmer
import pyarabic.arabrepr
from tashaphyne.stemming import ArabicLightStemmer
import re


stop_words = get_stop_words('ar')
stop_words = get_stop_words('arabic')

def tokenize_text(text):
    tokens = word_tokenize(text) 
    tokens = [token.strip() for token in tokens]
    return tokens

def remove_stopwords(text):
    tokens = tokenize_text(text)
    filtered_tokens = [token for token in tokens if token not in stop_words]
    filtered_text = ' '.join(filtered_tokens)    
    return filtered_text

def stem_text(text, stem):
    Is = ISRIStemmer()
    Al = ArabicLightStemmer()
    
    tokens = tokenize_text(text)
    if stem == 'Is':
        filtered_tokens = [Is.stem(token) for token in tokens]
    if stem == 'Al':
        filtered_tokens = [Al.light_stem(token) for token in tokens]
        
    filtered_text = ' '.join(filtered_tokens)
    return filtered_text

def lemmatize_text(text):
    arepr = pyarabic.arabrepr.ArabicRepr()
    repr = arepr.repr
    lemmer = qalsadi.lemmatizer.Lemmatizer()
    text = ' '.join(lemmer.lemmatize_text(text))
    return text

arabic_diacritics = re.compile("""
                             ّ    | # Tashdid
                             َ    | # Fatha
                             ً    | # Tanwin Fath
                             ُ    | # Damma
                             ٌ    | # Tanwin Damm
                             ِ    | # Kasra
                             ٍ    | # Tanwin Kasr
                             ْ    | # Sukun
                             ـ     # Tatwil/Kashida
                         """, re.VERBOSE)
def remove_diacritics(text):
    text = re.sub(arabic_diacritics, '', text)
    return text

def normalize_arabic(text):
    text = re.sub("[إأآا]", "ا", text)
    text = re.sub("ى", "ي", text)
    text = re.sub("ؤ", "ء", text)
    text = re.sub("ئ", "ء", text)
    text = re.sub("ة", "ه", text)
    text = re.sub("گ", "ك", text)
    return text
    
def normalize_corpus(corpus, stopwords_removal=True,
                     stemming=True,stem ='Is', diacritics_removal=True, 
                     normalization=True, lemmatization =True):
    
    normalized_corpus = []
        
    for doc in corpus:
        
        if normalization:
            doc = normalize_arabic(doc)
            
        if lemmatization:
            doc = lemmatize_text(doc)
            
        if stemming:
            doc = stem_text(doc,stem)
            
        if diacritics_removal:
            doc = remove_diacritics(doc)
            
        if stopwords_removal:
            doc = remove_stopwords(doc).strip()
            
        normalized_corpus.append(doc)
        
    return normalized_corpus
    




# In[ ]:




