from flask import Flask,request
from joblib import load
import numpy as np

app = Flask(__name__)

@app.route("/compare",methods=["POST"])
def COMPARE_DIGITS_PREDICTED():
    json = request.get_json()
    image = [float(i) for i in  json["input1"]]

    image2 = [float(i) for i in json["input2"]]
    model = load("models/svm_gamma:0.0001_C:10.joblib")
    
    image, image2 = np.array(image).reshape(-1,64), np.array(image2).reshape(-1,64)
    
    prediction_1, prediction_2 =  model.predict(image), model.predict(image2)

    return str(prediction_1 == prediction_2)


@app.route("/predict/<to_check>",methods=["POST"])
def PREDICTED_MODELS(to_check):
    json = request.get_json()
    image = [float(i) for i in json["input"]]
    trans = load('./models/transforms.joblib')
    image = trans.transform(np.array(image).reshape(-1,64))

    models_dict = {
        "svm":f'./models/svm_gamma:1_C:10.joblib',
        "tree:f'./models/M23CSA014_tree_max_depth:10.joblib',
        "lr":f'./models/M23CSA014_lr_solver:liblinear.joblib'
    }
    
    if to_check == "svm":
        model = load(models_dict['svm])
    if to_check == 'tree':
        model = load(models_dict['tree'])
    if to_check == 'lr':
        model = load(models_dict["lr"])
    
    return str(model.predict(image)[0])
