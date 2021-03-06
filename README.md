# Disaster Response Pipeline Project

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

### Files details
1. models
	- train_classifier is a pipeline that gets the data from the database and trains a multioutput random forest with a gridsearch for min_sample_split
    - classiifer.pkl is the saved model
    
2. data
	- process_data.py takes the data from teh csv files disaster_categories.cav and disaster_messages.csv, cleans the data and store it in 
    
3. app
	- run.py is using Flask and Plotly to display a webpage where we can enter messages and get their category using the ML model trained previously
