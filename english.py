#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!/usr/bin/env python
# coding: utf-8

# In[1]:


# # Import necessary dependencies
import re
import nltk
import spacy
import string
import unicodedata
from nltk.tokenize import word_tokenize
from bs4 import BeautifulSoup
from nltk.stem import PorterStemmer, SnowballStemmer
from nltk.stem.lancaster import LancasterStemmer
nlp = spacy.load('en_core_web_sm', disable=['ner', 'parser'])

stopword_list = nltk.corpus.stopwords.words('english')

def tokenize_text(text):
    tokens = word_tokenize(text) 
    tokens = [token.strip() for token in tokens]
    return tokens


# # Removing accented characters
def remove_accented_chars(text):
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    return text


# Lemmatizing text based on Spacy
def lemmatize_text(text):
    text = nlp(text)
    text = ' '.join([word.lemma_ for word in text])
    # text = ' '.join([word.lemma_ if word.lemma_ != '-PRON-' else word.text for word in text])
    return text


# # Removing Stopwords
def remove_stopwords(text, is_lower_case=False):
    tokens = tokenize_text(text)
    if is_lower_case:
        filtered_tokens = [token for token in tokens if token not in stopword_list]
    else:
        filtered_tokens = [token for token in tokens if token.lower() not in stopword_list]
    filtered_text = ' '.join(filtered_tokens)    
    return filtered_text

# # Stemming text : method used depends on input
def stem_text(text, stem):
    ps = PorterStemmer()
    ls = LancasterStemmer()
    ss = SnowballStemmer("english")
    
    tokens = tokenize_text(text)
    if stem == "ps": # porter stemming
        tokens = [ps.stem(token) for token in tokens]
    elif stem == "ls": # lancaster stemming
        tokens = [ls.stem(token) for token in tokens]
    elif stem == "ss": # snowball stemming
        tokens = [ss.stem(token) for token in tokens]
    text = ' '.join(tokens)
    return text


# # Normalize text corpus - tying it all together
def normalize_corpus(corpus, accented_char_removal=True, text_lower_case=True, 
                     text_lemmatization=True, stopword_removal=True,text_stemming=True,stem='ps'):
    
    normalized_corpus = []
    
    for doc in corpus:
        
        if accented_char_removal:
            doc = remove_accented_chars(doc)

        if stopword_removal:
            doc = remove_stopwords(doc, is_lower_case=text_lower_case)
        
        if text_lemmatization:
            doc = lemmatize_text(doc) 
        
        if text_stemming:
            doc = stem_text(doc,stem)
            
        if text_lower_case:
            doc = doc.lower()
            
        normalized_corpus.append(doc)
        
    return normalized_corpus

