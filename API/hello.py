
from flask import Flask,request
from sklearn import svm
from joblib import load
from flask import Flask

app = Flask(__name__)

@app.route("/<name>")
def hello_world(name):
    return "<p>Hello, World!</p>" + name
@app.route("/<name>/<a>/<b>")
def sum_of_numbers(name,a,b):
    return name  + " your sum is " + str(int(a) + int(b))

@app.route("/predict",methods=["POST"])
def digit_prediction():
    js = request.get_json()
    img_1 = js["input1"]
    img_1 = [float(i) for i in img_1]

    img_2 = js["input2"]_
    img_2 = [float(i) for i in img_2]

    model = load("models/svm_gamma:0.0001_C:10.joblib")
    import numpy as np
    img_1 = np.array(img_1).reshape(-1,64)
    img_2 = np.array(img_2).reshape(-1,64)
    pred_1 = model.predict(img_1)
    pred_2 = model.predict(img_2)
    if pred_1 == pred_2:
        return "TRUE"
    return "FALSE"
    return name  + " your sum is " + str(int(a) + int(b))
