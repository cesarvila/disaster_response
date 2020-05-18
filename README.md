# Disaster Response

### Date Created and Autor
`César Vila`
Dated on **11th May 2020**

### Project Title

**Disaster Response** Application

### Description and Motivation of the project

Using my *skills* learned at **Udacity**, an online educational organization, I
 have created this project to `analyze and classify messages` inspired by
 **Appen**, a company which provides or improves *data* used for the development
  of `machine learning` and `artificial intelligence` products.

In case of a *disaster*, people in trouble might post messages in social media 
asking for help. There are also other ways to seek for help like calling to
emergency or any other way. The aim of the project is the classification of 
the messages in order to **provide the proper support to the people**.

### Summary of the result

Appen provides a couple of csv with messages and categories. I have clean those
files and save into my Database `DisasterResponse.db`.
You will be able to create my database in your computer by following these
instructions:
 - Fork this repository, clone it in your PC using git.
 - go in your terminal to the folder data
 - run the following `python process_data.py messages.csv categories.csv DisasterResponse.db`
 
The name of the created table is CategorizedMessages

Based on that table, I created a *model* by tokenizing and vectorizing the messages.

The model created is a pipeline with:
 - **TfidfVectorizer** which Convert a collection of raw documents to a matrix
 of TF-IDF features
 - **MultiOutputClassifier**, a multi target classification of a **RandomForestClassifier**, 
 a meta estimator that fits a number of decision tree classifiers on various sub-samples of
 the dataset and uses averaging to improve the predictive accuracy and control over-fitting.
 
The parameters used in my model has been obtained through a search within the **GridSearchCV** 
which is an exhaustive search over specified parameter values for an estimator.

I have obtained a good accuracy, recall and f1 score. Almost **95%**

To get the model in your computer, you can follow the instructions below, but take into
consideration that the model is heavy, it will be a file named `classifier.pkl` around
800 MB file size:
 - you have already forked and cloned this repo
 - go in your terminal to the folder models
 - run the following `python train_classifier.py --database_filename ../data/DisasterResponse.db --model_pickle_filename classifier.pkl --grid_search_cv`

Now, using this model, you will be able to classify messages. 

A great way to visualize this, is throughout a webabb. You can access to my webapp by:
 - you have already forked and cloned this repo
 - go in your terminal to the folder app
 - run the following `python run.py`
 - go to http://0.0.0.0:3001/  (http://localhost:3001/)

As you can see, firstly is showed 3 insights of the messages provided by Appen, quite interesting,
**do you agree?**

Now, try to type some messages, like *I am hungry, please get me some food* or
*It is coming a big storm*, you can try with any message and you will see how the message is clasify
whithin one of the 36 categories showed.

### Files used

The used files are: 
 1) `data/process_data.py` as the script to clean and create my 
datebase.
 2) `models/train_classifier.py` as the script to create the model
 3) `app/run.py`as the script to run the webapp

### Libraries used:
 - **numpy** and **pandas** to *create dataframes* and work with them
 - **sys**
 - **sqlalchemy**
 - **re**
 - **argparse**
 - **nltk** (`ltk.download(['punkt', 'wordnet', 'stopwords', 'averaged_perceptron_tagger'])`)
 - **pickle**
 - **json**
 - **flask**
 - **plotly**
 - **joblib**
 - **sklearn** for the *model* 

### Credits

 * Udacity
 * Appen (Figure Eight)
 * Stackoverflow
 * scikit-learn
 * w3schools
 
