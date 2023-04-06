from flask import Flask, request, jsonify
import pickle
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn import preprocessing
from keras.utils import to_categorical
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from tensorflow.python.keras import layers

app = Flask(__name__)

disease_prediction = pickle.load(open('smote_xgboost_model.pkl','rb'))
#heart_disease_prediction = pickle.load(open('finalized_detection_model.pkl','rb'))

@app.route('/')
def home():
    return "hello world"

@app.route('/predict',methods = ["POST"])
def predict_dis():
    q = request.form.get('symptoms').split(',')
    age = int(request.form.get('age'))
    gender = request.form.get('gender')
    #d = request.form.get('data').split(",")
    #hd = [float(i) for i in d]
    disease = ''
    query = []
    count = 0
    data = pd.read_csv("./Symptom-severity.csv")
    data1 = pd.read_csv("./Disease_Label.csv")
    for i in range(len(q)):
        for j in range(len(data)):
            if q[i] in data.iloc[j]["Symptom"]:
                query.append(data.iloc[j]["weight"])
                count +=1
    for k in range(17-count):
        query.append(0)
    input_query = np.array([query])
    #input_hd = np.array([hd])
    result = disease_prediction.predict(input_query)
    #heart_d = heart_disease_prediction.predict(input_hd)
    for j in range(len(data1)):
        if result[0] == int(data1.iloc[j]["Disease_Labels"]):
            disease = data1.iloc[j]["Disease"]
            break
    #drug = recommendation(age , disease ,gender)
    return jsonify({'disease':str(disease)})

if __name__ == '__main__':
    app.run(debug=True)