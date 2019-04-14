import csv

from flask import Flask,render_template, request, current_app
from src.algorithms.knn.knn import pre_process_data_optimised, predict, load_file, preprocess_input
import pandas as pd

app = Flask(__name__)
# df = load_file('/Users/hao/PycharmProject/COMP9321-project/data/heart_disease.csv')
# df = pre_process_data_optimised(df)


def make_prediction(row):
    df = load_file(current_app.root_path + '/data/heart_disease.csv')
    row = pd.DataFrame([row], columns=['a'+str(i) for i in range(1,14)])
    row = preprocess_input(row, df)
    df = pre_process_data_optimised(df)
    return predict(row.loc[0][:], df, 3)


@app.route('/')
@app.route('/visualisation')
def visualisation():
    return render_template("visualisation.html")


@app.route('/important-attributes')
def important_attributes():
    return render_template("important-attributes.html")


@app.route('/prediction', methods=["POST", "GET"])
def prediction():
    if request.method == "POST":
        try:
            row = request.form["row"]
            row = row.split(",")
            row = list(map(lambda x: float(x), row))
            print(row)
            result = int(make_prediction(row))

            return render_template("prediction.html", result=str(result), row=request.form["row"])
        except:
            return render_template("prediction.html", error="Invalid number of attributes in input.", row=request.form["row"])

    return render_template("prediction.html")


@app.route('/bonus')
def bonus():
    return render_template("bonus.html")


if __name__ == '__main__':
    app.run(debug=True, port=5433)
