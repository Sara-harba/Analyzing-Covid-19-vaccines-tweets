#!/usr/bin/env python
# coding: utf-8

# In[2]:


from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import ComplementNB
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import roc_auc_score, confusion_matrix, roc_curve
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score

#Base Model
def naive_bayes_model1(train_vector, train_labels):

    mnb = MultinomialNB(alpha=1)
    
    mnb_train_score = np.mean(cross_val_score(mnb, train_vector, train_labels, cv=5, scoring='roc_auc'))

    return mnb_train_score


# Naïve Bayes ComplementNB
def naive_bayes_cnb(train_vector, train_labels,alpha):

    cnb = ComplementNB(alpha=alpha)
    
    auc_score = np.mean(cross_val_score(cnb, train_vector, train_labels, cv=5, scoring='roc_auc'))
 
    return auc_score

# Logistic Regression
def logistic_model(train_vector, train_labels, C, max_iter):
    
    lr = LogisticRegression(penalty='l2', max_iter=max_iter, C=C, random_state=42)
    
    auc_score = np.mean(cross_val_score(lr, train_vector, train_labels, cv=5, scoring='roc_auc'))

    return auc_score
        

#KNN
def knn_model(train_vector, train_labels, n_neighbors):
    
    knn = KNeighborsClassifier(n_neighbors = n_neighbors)

    auc_score = np.mean(cross_val_score(knn, train_vector, train_labels, cv=5, scoring='roc_auc'))

    return auc_score

#Decision Tree
def decision_tree_model(train_vector, train_labels,max_depth):
    
    clf = DecisionTreeClassifier(max_depth= max_depth)
    
    auc_score = np.mean(cross_val_score(clf, train_vector, train_labels, cv=5, scoring='roc_auc'))

    return auc_score


# SVM with Stochastic Gradient Descent
def svm_sgd_model(train_vector, train_labels, max_iter, alpha):
    
    svm_sgd = SGDClassifier(loss='modified_huber', penalty='l2', max_iter=max_iter, random_state=42)

    auc_score = np.mean(cross_val_score(svm_sgd, train_vector, train_labels, cv=5, scoring='roc_auc'))

    return auc_score

