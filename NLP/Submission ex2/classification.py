#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import testsets
import evaluation
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import FeatureUnion, Pipeline
import textPreprocessor as tp
import time
import nltk
import numpy as np
import twokenize
from sklearn.preprocessing import FunctionTransformer
start_time = time.time()

""" Extracts tweet length """
def get_tweet_length(text):
    return len(text)

""" Functions to help pipeline work """
# Taken from https://ryan-cranfill.github.io/sentiment-pipeline-sklearn-4/ 
def pipelinize(function, active=True):
    def list_comprehend_a_function(list_or_series, active=True):
        if active:
            return [function(i) for i in list_or_series]
        else: 
            return list_or_series
    return FunctionTransformer(list_comprehend_a_function, validate=False, kw_args={'active':active})

def reshape_a_feature_column(series):
    return np.reshape(np.asarray(series), (len(series), 1))

def pipelinize_feature(function, active=True):
    def list_comprehend_a_function(list_or_series, active=True):
        if active:
            processed = [function(i) for i in list_or_series]
            processed = reshape_a_feature_column(processed)
            return processed
        else:
            return reshape_a_feature_column(np.zeros(len(list_or_series)))

""" Tokenisers """
tokenizer = nltk.casual.TweetTokenizer(preserve_case=False, reduce_len=True)
tokenizer2 = twokenize.tokenize

""" Text Preprocessor """
text_preprocessor = tp.pipeproc

""" Opens and stores training data in data frame using pandas """
df = pd.read_csv("twitter-training-data.txt", sep="\t", names = ["ID", "Sentiment", "Tweet"], dtype={"ID": str})

""" Training setup """
X, y = df["Tweet"], df["Sentiment"]

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, train_size=0.99)

count_vect = CountVectorizer(tokenizer=tokenizer2)  # feature extracter 

""" Classifier training """
for classifier in ['myclassifier1', 'myclassifier2', 'myclassifier3']: 
    if classifier == 'myclassifier1':
        print('Training ' + classifier)

        classif = LogisticRegression()
        
        pipeline = Pipeline([
                ('text_preproc', pipelinize(text_preprocessor, active=True)),
                ('features', FeatureUnion([
                            ('vectorizer', count_vect),
                            ('post_length', pipelinize_feature(get_tweet_length, active=True))
                        ])),
                ('classifier', classif)
            ])
        
        pipeline.fit(X_train, y_train)
        
        preds = pipeline.predict(X_test)
        
        print("training accuracy:", np.mean(preds == y_test), "\n")  # prints training accuracy
        
        pipeline = Pipeline([('vectorizer', count_vect), ('classifier', classif)])
        
        pipeline.fit(X_train, y_train)
        
    elif classifier == 'myclassifier2':
        print('Training ' + classifier)
        
        classif = MultinomialNB()
        
        pipeline = Pipeline([
                ('text_preproc', pipelinize(text_preprocessor, active=True)),
                ('features', FeatureUnion([
                            ('vectorizer', count_vect),
                            ('post_length', pipelinize_feature(get_tweet_length, active=True))
                        ])),
                ('classifier', classif)
            ])
        
        pipeline.fit(X_train, y_train)
        
        preds = pipeline.predict(X_test)
        
        print("training accuracy:", np.mean(preds == y_test), "\n")
            
    
    
    elif classifier == 'myclassifier3':
        print('Training ' + classifier)
        
        classif = SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, random_state=42,max_iter=5, tol=None)
        
        pipeline = Pipeline([
                ('text_preproc', pipelinize(text_preprocessor, active=True)),
                ('features', FeatureUnion([
                            ('vectorizer', count_vect),
                            ('post_length', pipelinize_feature(get_tweet_length, active=True))
                        ])),
                ('classifier', classif)
            ])
        
        pipeline.fit(X_train, y_train)
        
        preds = pipeline.predict(X_test)
        
        print("training accuracy:", np.mean(preds == y_test), "\n")
    

    for testset in testsets.testsets:
        
        """ Opens test data """
        dft = pd.read_csv(testset, sep="\t", names = ["ID", "Sentiment", "Tweet"], dtype={"ID": str})
        
        dft_id = list(dft["ID"])
        
        dft_sent = list(dft["Sentiment"])
        
        dft_act = dict(zip(dft_id, dft_sent))  # stores actual result
        
        preds = pipeline.predict(dft["Tweet"])  # predictions
        
        predictions = dict(zip(dft_id, preds))  # formats preds into dict for evaluation.py 
        
        print("Overall", "accuracy" + ":", np.mean(preds == dft["Sentiment"]))
        
        evaluation.evaluate(predictions, testset, classifier)

        evaluation.confusion(predictions, testset, classifier)
                
    print("\n", "="*35)
    
print("\n", "--- {} seconds ---".format(time.time() - start_time))

