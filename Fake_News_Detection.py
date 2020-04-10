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
