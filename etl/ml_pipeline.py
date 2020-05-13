# -*- coding: utf-8 -*-
"""
Created on Tue May 12 20:55:54 2020

@author: cvila
"""


# import libraries
from sqlalchemy import create_engine

import pandas as pd
import numpy as np

import re

import nltk
nltk.download(['punkt', 'wordnet', 'stopwords', 'averaged_perceptron_tagger'])

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.stem import WordNetLemmatizer 
from nltk.stem.porter import PorterStemmer

import pickle

from sklearn.model_selection import train_test_split #Split arrays or matrices into random train and test subsets
#The sklearn.pipeline module implements utilities to build a composite estimator, as a chain of transforms and estimators.
from sklearn.pipeline import Pipeline #Pipeline(steps[, memory, verbose]) Pipeline of transforms with a final estimator.
from sklearn.pipeline import FeatureUnion #FeatureUnion(transformer_list[, …])Concatenates results of multiple transformer objects.
from sklearn.feature_extraction.text import CountVectorizer #Convert a collection of text documents to a matrix of token counts
from sklearn.feature_extraction.text import TfidfVectorizer #Convert a collection of raw documents to a matrix of TF-IDF features.
from sklearn.feature_extraction.text import TfidfTransformer#Transform a count matrix to a normalized tf or tf-idf representation
from sklearn.metrics import confusion_matrix, f1_score, recall_score, precision_score, accuracy_score
from sklearn.ensemble import RandomForestClassifier #A random forest classifier
from sklearn.multioutput import MultiOutputClassifier  #Multi target classification
from sklearn.metrics import classification_report, precision_recall_fscore_support
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV#GridSearchCV(estimator, …)Exhaustive search over specified parameter values for an estimator.

# load data from database

db_name = 'DisasterResponse.db'
table_name = 'CategorizedMessages'
model_pickle = 'trained_classifier.pkl'


engine = create_engine('sqlite:///' + db_name)
df = pd.read_sql(table_name, con = engine)
df.head()
X = df['message']
Y = df.iloc[:,4:]
category_names = Y.columns.tolist()
stop_words = set(stopwords.words('english'))
url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
not_alphanum = r"[^a-zA-Z0-9]"
not_alphanum_ = r'\W'
numbers_r = '\d'
def tokenize(text):
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, 'urlplaceholder')
    text = word_tokenize(re.sub(not_alphanum,' ',text).lower().strip())
    lemmatizer = WordNetLemmatizer()
    clean_words = []
    for word in text:
        clean_word = lemmatizer.lemmatize(word, pos='v')
        clean_words.append(clean_word)
    return clean_words
def get_pipeline(estimator=RandomForestClassifier(random_state=42, n_jobs =-1), tokenize_function=tokenize):
    return Pipeline([
        ('tfidfvect', TfidfVectorizer(tokenizer=tokenize_function, stop_words='english')),
        ('moc', MultiOutputClassifier(estimator))
    ])
pipe1 = get_pipeline()
pipe1.get_params().keys()
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
def get_model(pipeline):
    return pipeline.fit(X_train, y_train)

model1 = get_model(pipe1)

def model_results(model):
    y_pred = pd.DataFrame(model.predict(X_test), columns = category_names)
    tot_acc = 0
    tot_f1 = 0
    k = 0 
    for category in category_names:
        k += 1
        #Each class label will be presented with a classification report
        print('{}. For the category "{}"'.format(k, category), 'model\'s accuracy is: {:.2f}'.format(accuracy_score(y_test.loc[:, category], y_pred.loc[:, category])),'\nand the Clasification Report is:\n',
              classification_report(y_test.loc[:, category], y_pred.loc[:, category]),'\n')
        tot_acc += accuracy_score(y_test.loc[:, category], y_pred.loc[:, category])
        tot_f1 += precision_recall_fscore_support(y_test.loc[:, category], y_pred.loc[:, category], average = 'weighted')[2]
    print('The average accuracy score for all labels is {},\
          and the average f1-score for all labels is {}'\
              .format(round(tot_acc/len(category_names),3),round(tot_f1/len(category_names),3)))

      
model_results(model1)
 
parameters_1 = {
        'tfidfvect__ngram_range': [(1, 1), (1, 2)],
        'tfidfvect__max_df': [0.75, 1.0],
        'tfidfvect__max_features': [2000, 5000],
        'moc__estimator__n_estimators': [10, 15],
        'moc__estimator__max_depth': [None, 15],
        'moc__estimator__min_samples_split': [2, 5]
        }

cv = GridSearchCV(model1, param_grid=parameters_1, cv=2)
model2 = cv.fit(X_train, y_train)
print('Best Parameters: ', model2.best_params_)
model_results(model2)
class StartingVerbExtractor(BaseEstimator, TransformerMixin):

    def starting_verb(self, text):
        sentence_list = nltk.sent_tokenize(text)
        for sentence in sentence_list:
            pos_tags = nltk.pos_tag(tokenize(sentence))
            first_word, first_tag = pos_tags[0]
            if first_tag in ['VB', 'VBP'] or first_word == 'RT':
                return True
        return False

    def fit(self, x, y=None):
        return self

    def transform(self, X):
        X_tagged = pd.Series(X).apply(self.starting_verb)
        return pd.DataFrame(X_tagged)

pipe3 = Pipeline([
    ('features', FeatureUnion([
        ('text_pipeline', Pipeline([
            ('vect', CountVectorizer(tokenizer=tokenize))
            , ('tfidf', TfidfTransformer())]))
        , ('starting_verb', StartingVerbExtractor())]))
    , ('clf', RandomForestClassifier())])

pipe3.get_params()
model3 = pipe3.fit(X_train, y_train)
model_results(model3)

pickle.dump(model3, open(model_pickle, 'wb'))
model = pickle.load(open(model_pickle, 'rb'))

category_predicted = model.predict(['I am thirsty, please bring me some water'])[0]