import joblib
from missDT import missingDataTransformer
from flask import Flask, request, jsonify
import pandas as pd

# create the flask app
app = Flask(__name__)


# API ROUTINNG
@app.route('/predict',methods = ['POST'])
def predict():


    # get JSON data sent
    feat_data = request.json
    # covert the json to dataframe
    df = pd.DataFrame(feat_data)
    # match the column names
    df = df.reindex(columns = col_names)

    # get the prediction
    predictions = list(model.predict(df))

    # return jsons version of prediction
    return jsonify({'prediction': str(predictions)})


if __name__ == '__main__':

    # load the model and feature
    model = joblib.load('HRR.joblib')
    col_names = joblib.load('col_names.joblib')

    app.run(debug=True)

    #[{"number_project":2, "average_montly_hours":142, "time_spend_company":3, "Work_accident":0, "promotion_last_5years":0, "department":"technical", "salary":"low", "satisfaction_level":0.39, "last_evaluation":0.47}]

