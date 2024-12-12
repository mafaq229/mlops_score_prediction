from flask import Flask, request, render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.pipeline.predict_pipeline import CustomData, PredictPipeline


application = Flask(__name__)

app = application

## Route for a home page
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predictdata", methods=["GET", "POST"])
def predict_datapoint():
    if request.method == "GET":
        return render_template("home.html")
    else:
        if request.is_json:
            req_data = request.get_json()
        else:
            req_data = request.form.to_dict()
            
        data = CustomData(
            gender=req_data.get('gender'),
            race_ethnicity=req_data.get('ethnicity'),
            parental_level_of_education=req_data.get('parental_level_of_education'),
            lunch=req_data.get('lunch'),
            test_preparation_course=req_data.get('test_preparation_course'),
            reading_score=float(req_data.get('writing_score')),
            writing_score=float(req_data.get('reading_score'))
        )
        print(request.form.to_dict())
        pred_df = data.get_data_as_frame()
        print(pred_df)
        
        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)
        return render_template("home.html", results=results[0])
    

if __name__ == "__main__":
    app.run(host="0.0.0.0")
