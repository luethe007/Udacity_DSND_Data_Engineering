import json
import plotly
import pickle
import pandas as pd

# pip install xgboost via terminal
from xgboost import XGBClassifier
from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar, Histogram
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.externals import joblib
from sqlalchemy import create_engine

import sys
sys.path.append('../models/')
from train_classifier import calculateTextLength, tokenize

app = Flask(__name__)

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('df', engine)

# load model
model = pickle.load(open('../models/classifier.pkl', 'rb'))

# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    cat_frequency = df[df.columns[4:]].sum().sort_values(ascending=False)[:10]
    cats = cat_frequency.index
    
    msg_len = df.message.apply(lambda x: len(x.split()))
    msg_len = msg_len[~((msg_len-msg_len.mean()).abs() > 3*msg_len.std())]
    # create visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=cats,
                    y=cat_frequency,
                    marker=dict(
                         color='#031c57'
                    )
                )
            ],

            'layout': {
                'title': 'Top 10 Message Categories',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Category"
                }
            }
        },
        {
            'data': [
                Histogram(
                    x=msg_len,
                    marker=dict(
                         color='#031c57'
                    )
                )
            ],

            'layout': {
                'title': 'Distribution of Message Lengths',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Words/Message"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts,
                    marker=dict(
                         color='#031c57'
                    )
                )
            ],

            'layout': {
                'title': 'Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)

# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )

def main():
    
    app.run(host='0.0.0.0', port=3001, debug=True)

if __name__ == '__main__':
    main()