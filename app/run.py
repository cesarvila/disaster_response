# python run.py
import json
import plotly
import pandas as pd
import plotly.express as px
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from flask import Flask
from flask import render_template, request
from plotly.graph_objs import Bar
from plotly.graph_objects import Sunburst
import joblib
from sqlalchemy import create_engine
import plotly.graph_objects as go

app = Flask(__name__, template_folder='template')

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('CategorizedMessages', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():

    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)





    df1 = df.drop(['id', 'message', 'original'], axis=1)
    dfmelt = pd.melt(df1, id_vars=['genre'], var_name = 'categories', value_name='count')
    df_group = dfmelt.groupby(['genre', 'categories']).sum()
    categories = []
    genres = []
    for i in range(df_group.shape[0]):
        genres.append(df_group.index[i][0])
        categories.append(df_group.index[i][1])
    values = df_group['count'].tolist()
    d = {'genres': genres, 'categories': categories, 'values': values}
    df_plot = pd.DataFrame(d)
    df_plot.sort_values(by='values', ascending=False, inplace=True)
    df_group1 = df1.groupby(['genre']).sum()
    fig = px.sunburst(df_plot, path=['genres', 'categories'], values='values')
    #fig = px.sunburst(df_plot, path=['genres', 'categories'], values='values')
    # create visuals
    # TODO: Below is an example - modify to create your own visuals

    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts,
                    text=[x for x in genre_names],
                    marker=dict(
                        color='rgb(158,202,225)',
                        line=dict(
                            color='rgb(8,48,107)',
                            width=1.5,
                        )
                    ),
                    opacity=0.8
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
        {
            'data': [
                Sunburst(
                    labels=fig['data'][0]['labels'].tolist(),
                    parents=fig['data'][0]['parents'].tolist(),
                    values=fig['data'][0]['values'].tolist()
                )
            ],
            'layout': {
                'margin' : {
                    't': 35,
                    'l': 2,
                    'r': 2,
                    'b': 2
                },
                'title': 'Sunburst graphic'


            }
        },
        {
            'data': [
                Bar(
                    name='News',
                    x=df_group1.columns.tolist(),
                    y=df_group1.loc['news'].tolist(),
                    marker_color='#e03c31'
                ),
                Bar(
                    name='Direct',
                    x=df_group1.columns.tolist(),
                    y=df_group1.loc['direct'].tolist(),
                    marker_color='#9AA5AF'
                ),
                Bar(
                    name='Social',
                    x=df_group1.columns.tolist(),
                    y=df_group1.loc['social'].tolist(),
                    marker_color='#00acee'
                )
            ],
            'layout': {
                'barmode': 'stack',
                'title': 'Distribution of Message Categories',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Category",
                    'tickangle': 30,
                    'tickfont': {
                        'size': 10
                    }
                }
            }
        }
    ]
    #graphs.append(dict(data=px.sunburst(df_plot, path=['genres', 'categories'], values='values'))

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
