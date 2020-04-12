# -*- coding: utf-8 -*-
"""
1.Name- Arun Yadav
 Reg.id-11711181
2.Name-Neeraj Singh
 Reg.id-11707098
3.Name-saumy kumar kushwaha
"""
"""
Spyder Editor

This is a temporary script file.
"""
# Dataset Downloaded from Kaggle.com
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
df=pd.read_csv('C:\\Users\\LENOVO\\Downloads\\news.csv')
#Get shape and head
df.shape
df.head()
#Get the labels
labels=df.label
labels.head()
#Split the dataset
x_train,x_test,y_train,y_test=train_test_split(df['text'], labels, test_size=0.2, random_state=7)
#Initialize a TfidfVectorizer.For the stopping word
tfidf_vectorizer=TfidfVectorizer(stop_words='english', max_df=0.7)
Classifier=[PassiveAggressiveClassifier,DecisionTreeClassifier,RandomForestClassifier,LinearSVC,LogisticRegression]
Name=['PassiveAggressiveClassifier','DecisionTreeClassifier','RandomForestClassifier','LinearSVC','LogisticRegression']
# Fit and transform train set, transform test set
tfidf_train=tfidf_vectorizer.fit_transform(x_train) 
tfidf_test=tfidf_vectorizer.transform(x_test)
for i in range(0,5):
    if(i==0):
        p=Classifier[i](max_iter=50);
    else:
        p=Classifier[i]();
    p.fit(tfidf_train,y_train)
#DataFlair - Predict on the test set and calculate accuracy
    y_pred=p.predict(tfidf_test)
    score=accuracy_score(y_test,y_pred)
    print('Accuracy with '+Name[i]+'Algorithm is:'+str(round(score*100,2)))

