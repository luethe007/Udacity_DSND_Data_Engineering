# import libraries
import sys
import pandas as pd
import numpy as np
import re
import pickle
from sqlalchemy import create_engine
from workspace_utils import active_session

import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.multioutput import MultiOutputClassifier
# pip install xgboost via terminal
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.base import BaseEstimator, TransformerMixin

def load_data(database_filepath):
    '''
    Read the data from a SQLIte database and return features and labels
    '''
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_query('SELECT * FROM df', engine)
    X = df['message'].values
    output_columns = df.iloc[:,4:].columns
    y = df.iloc[:,4:].values
    return X, y, output_columns
    
def tokenize(text):
    '''
    Case normalize, lemmatize and tokenize text
    '''
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    clean_tokens = []
    for tok in tokens:
        clean_tok = re.sub(r'[^\w\s]','',tok)
        clean_tok = lemmatizer.lemmatize(clean_tok).lower().strip()
        if clean_tok:
            clean_tokens.append(clean_tok)
    return clean_tokens

class calculateTextLength(BaseEstimator, TransformerMixin):
    '''
    Custom transformer to calculate the text length
    '''
    def fit(self, x, y=None):
        return self

    def transform(self, X):
        X_tagged = pd.Series(X).apply(len)
        return pd.DataFrame(X_tagged)

def build_model():
    '''
    Process text and perform multi-output classification on 36 categories
    '''
    pipeline = Pipeline([
            ('features', FeatureUnion([
                ('text_pipeline', Pipeline([
                    ('vect', CountVectorizer(tokenizer=tokenize)),
                    ('tfidf', TfidfTransformer()),
                ])),
                ('text_length', calculateTextLength())
            ])),
            ('clf', MultiOutputClassifier(XGBClassifier(learning_rate=0.1, scale_pos_weight=1, seed=27)))
        ])
    # only the best parameters are set to reduce training time
    parameters = {
        'clf__estimator__max_depth': [10],
        'clf__estimator__min_child_weight': [3],
        'clf__estimator__gamma': [0.0],
        'clf__estimator__subsample': [0.7],
        'clf__estimator__colsample_bytree' :[0.8],
        'clf__estimator__reg_alpha': [0.01],
        'clf__estimator__n_estimators': [200]
    }
    cv = GridSearchCV(pipeline, param_grid=parameters)
    return cv

def evaluate_model(model, X_test, Y_test, category_names):
    '''
    Print the F1 score, precision and recall for each category of the test set
    '''
    y_pred = model.predict(X_test)    
    i = 0
    for col in category_names:
        sliced_y_pred = np.array(y_pred)[:,i]
        sliced_y_test = np.array(Y_test)[:,i]
        precision = precision_score(sliced_y_test, sliced_y_pred, labels=np.unique(sliced_y_pred))
        recall = recall_score(sliced_y_test,sliced_y_pred)
        f1_sc = f1_score(sliced_y_test,sliced_y_pred, labels=np.unique(sliced_y_pred))
        accuracy = accuracy_score(sliced_y_test, sliced_y_pred)
        print(col)
        print('Precision: {}, Recall: {}, F1 Score {}, Accuracy: {}'.format(str(precision), str(recall), str(f1_sc), str(accuracy)))    
        print('\n')
        i+=1

def save_model(model, model_filepath):
    '''
    Store the trained model
    '''
    pickle.dump(model, open(model_filepath, "wb"))

def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        with active_session():     
            model.fit(X_train, Y_train)
  
        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)
        print('Trained model saved!')
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()