# Train the model and save it in a pickle file
# python train_classifier.py --database_filename ../data/DisasterResponse.db --model_pickle_filename classifier.pkl --grid_search_cv
# import packages

# import libraries
from sqlalchemy import create_engine

import pandas as pd

import re

import argparse

import nltk
nltk.download(['punkt', 'wordnet', 'stopwords', 'averaged_perceptron_tagger'])


from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer


import pickle

from sklearn.model_selection import train_test_split
 #Split arrays or matrices into random train and test subsets

"""The sklearn.pipeline module implements utilities to build a composite
 estimator, as a chain of transforms and estimators."""
from sklearn.pipeline import Pipeline
#Pipeline(steps[, memory, verbose]) Pipeline of transforms with a final estimator.

from sklearn.feature_extraction.text import TfidfVectorizer
#Convert a collection of raw documents to a matrix of TF-IDF features.

from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
 #Multi target classification

from sklearn.metrics import classification_report, precision_recall_fscore_support
from sklearn.model_selection import GridSearchCV
#GridSearchCV(estimator, …)Exhaustive search over specified parameter values for an estimator.


#names of my files
db_name = 'DisasterResponse.db'
table_name = 'CategorizedMessages'
model_pickle = 'trained_classifier.pkl'

#supporting vars for tokenize function
url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
not_alphanum = r"[^a-zA-Z0-9]"



def load_data(data_file):
    # read in file
    engine = create_engine('sqlite:///' + data_file)
    df = pd.read_sql_table(table_name, con = engine)
    X = df['message']
    Y = df.iloc[:, 4:]
    category_names = Y.columns.tolist()
    return X, Y, category_names

def tokenize(text):
    #first remove any website
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, 'urlplaceholder')
    #normalize, tokenize...
    text = word_tokenize(re.sub(not_alphanum,' ',text).lower().strip())
    lemmatizer = WordNetLemmatizer()
    clean_words = []
    #lemmatize
    for word in text:
        clean_word = lemmatizer.lemmatize(word, pos='v')
        clean_words.append(clean_word)
    return clean_words

def build_model(grid_search_cv):
    """I am creating this function with grid_search_cv desactivated by default
    since it takes 6 to 7 hours in my laptop to fit the model,
    I will use the best parameters got in my Jupyter nb as the default
    parameters in my pipeline"""
    # text processing and model pipeline
    pipe = Pipeline([
        ('tfidfvect', TfidfVectorizer(tokenizer=tokenize, stop_words='english',
         max_df= 0.75, max_features = 5000, ngram_range=(1, 2))),
        ('moc', MultiOutputClassifier(RandomForestClassifier(random_state=42,
         n_jobs =-1, max_depth = None, min_samples_split = 2,
          n_estimators= 100)))
    ])
    # define parameters for GridSearchCV
    if grid_search_cv == True:
        """In case I got time, I will use other parameters to iterate """
        print('Searching for best parameters...')
        parameters = {
        'moc__estimator__verbose': [0, 10],
        'tfidfvect__sublinear_tf': [False, True]
        }
        pipe = GridSearchCV(pipe, param_grid = parameters, cv = 2)
    return pipe

def model_results(model, X_test, Y_test, category_names):
    """This function prints the results of the model"""
    y_pred = pd.DataFrame(model.predict(X_test), columns = category_names)
    tot_acc = 0
    tot_f1 = 0
    k = 0
    n_categories = len(category_names)
    detail = ''
    for category in category_names:
        ascore = accuracy_score(Y_test.loc[:, category], y_pred.loc[:, category])
        creport = classification_report(Y_test.loc[:, category], y_pred.loc[:, category])
        prfssupport = precision_recall_fscore_support(Y_test.loc[:, category],
                            y_pred.loc[:, category], average = 'weighted')[2]
        k += 1
        #Each class label will be presented with a classification report
        detail += '\n {}. Category "{}"'.format(k, category)
        detail += ' model\'s accuracy is: {:.2f}'.format(ascore)
        detail += '\nIts Clasification Report is:\n'
        detail += creport
        tot_acc += ascore
        tot_f1 += prfssupport
    avg_acc = round(tot_acc/n_categories, 5)
    avg_f1 = round(tot_f1/n_categories, 5)
    print('The average accuracy score for all labels is {},\
 and the average f1-score for all labels is {}'.format(avg_acc,avg_f1))
    print('\nFor the details, see data below:\n{}'.format(detail))
    print('\nSummary:\
     \nAverage accuracy score {}\nAverage f1-score {}'.format(avg_acc,avg_f1))

def export_model(model, model_filename):
    '''
    Save in a pickle file the model
    Inputs:
        model: model to be saved
        model_filename (str): destination pickle filename
    '''
    pickle.dump(model, open(model_filename, 'wb'))

def train(database_name, model_pickle, grid_search_cv):
    # train test split
    X, Y, category_names = load_data(database_name)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2,
                                                        random_state=42)
    # build model
    print('Building model...')
    model = build_model(grid_search_cv)
    # fit model
    print('Training model...')
    model.fit(X_train, Y_train)
    # output model test results
    print('Evaluating model...')
    model_results(model, X_test, Y_test, category_names)

    print('Saving model...\n    Model: {}'.format(model_pickle))
    export_model(model, model_pickle)

    print('Trained model exported!\n Its name is {}'.format(model_pickle))

def parse_input_arguments():
    '''
    Parse the command line arguments
    Returns:
        db_name (str): database filename. Default value DATABASE_FILENAME
        model_pickle (str): pickle filename. Default value MODEL_PICKLE_FILENAME
        grid_search_cv (bool): If True perform grid search of the parameters
    '''
    parser = argparse.ArgumentParser(description = "Disaster Response Pipeline\
     Train Classifier")
    parser.add_argument('--database_filename', type = str,
                            default = db_name,
                            help = 'Database filename of the cleaned data')
    parser.add_argument('--model_pickle_filename', type = str,
                            default = model_pickle,
                            help = 'Pickle filename to save the model')
    parser.add_argument('--grid_search_cv', action = "store_true",
                            default = False,
                            help = 'Perform grid search of the parameters')
    args = parser.parse_args()
    #print(args)
    return args.database_filename, args.model_pickle_filename, args.grid_search_cv


if __name__ == '__main__':
    database_name, model_pickle, grid_search_cv = parse_input_arguments()
    train(database_name, model_pickle, grid_search_cv)
