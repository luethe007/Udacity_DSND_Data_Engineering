# Data Scientist Nanodegree
# Data Engineering
## Project: Disaster Response Pipeline

### Install
This project requires **Python 3.x** and the following Python libraries installed:

- [NumPy](http://www.numpy.org/)
- [Pandas](https://pandas.pydata.org/)
- [SQLAlchemy](https://www.sqlalchemy.org/)
- [nltk](https://www.nltk.org/)
- [scikit-learn](https://scikit-learn.org/stable/)
- [XGBoost](https://xgboost.readthedocs.io/en/latest/)
- [Plotly](https://plot.ly/)
- [Flask](https://palletsprojects.com/p/flask/).

### Code
The code has been created as part of the Udacity Data Scientist Nanodegree.

Files:
- process_data.py: takes the file paths of the two datasets and database, cleans the datasets, and stores the clean data into a SQLite database in the specified database file path
- train_classifier.py: takes the database file path and model file path, creates and trains a classifier, and stores the classifier into a pickle file to the specified model file path
- run.py: runs the Flask web app.

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

### Licensing, Authors, Acknowledgements<a name="licensing"></a>

Feel free to use the code here as you would like!
