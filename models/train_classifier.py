import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

import re
import nltk
from nltk.tokenize import word_tokenize
nltk.download('stopwords')
nltk.download(['punkt', 'wordnet'])
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV

from joblib import dump, load


def load_data(database_filepath):
    ''' 
    load data to databse
    return X, y, category_names
    '''
    engine = create_engine('sqlite:///'+ database_filepath)
    df = pd.read_sql_table('DisasterResponse.db', engine)
    X = df.iloc[:,1]
    y = df.iloc[:,4:]
    category_names = y.columns
    return X, y, category_names


def tokenize(text):
    '''
    tokenize text
    input: text
    return: tokens
    '''
    # case normalization
    text = text.lower()
    
    # Remove punctuation characters
    text = re.sub(r"[^a-zA-Z0-9]", " ", text) 
    
    # Split text into words using NLTK
    words = word_tokenize(text)
    
    # Remove stop words
    words = [w for w in words if w not in stopwords.words("english")]
    
    # Reduce words to their root form
    lemmed = [WordNetLemmatizer().lemmatize(w) for w in words]
    
    return lemmed


def build_model():
    '''
    build pipeline with random forest and gridsearch
    return: model
    '''
    #pipeline
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('mlt', MultiOutputClassifier(estimator=RandomForestClassifier(), n_jobs=-1))
    ])
    # gridsearch
    parameters = {
        'mlt__estimator': [RandomForestClassifier(min_samples_split=2), 
                           #RandomForestClassifier(min_samples_split=3),
                            #RandomForestClassifier(min_samples_split=4)
                          ]
    }

    model = GridSearchCV(pipeline, param_grid=parameters)
    
    return model


def evaluate_model(model, X_test, Y_test, category_names):
    '''
    evaluate model to get F1 score, accuracy, recall
    '''
    # predict
    y_pred = model.predict(X_test)
    
    # evaluate
    for col, cat in zip(range(0,y_pred.shape[1]), category_names):
        print(cat)
        print(classification_report(Y_test.iloc[:,col],y_pred[:,col]))


def save_model(model, model_filepath):
    dump(model, model_filepath)


def main():
    '''
    train and save the model
    '''
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()