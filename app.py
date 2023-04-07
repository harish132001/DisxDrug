from flask import Flask, request, jsonify
import pickle
import pandas as pd
import numpy as np
#import tensorflow as tf
from sklearn import preprocessing
from keras.utils import to_categorical
from sklearn.feature_extraction.text import TfidfVectorizer
#from sklearn.model_selection import train_test_split
#from tensorflow.python.keras import layers
from keras.saving.legacy.model_config import model_from_json
#from tensorflow.python.keras.saving.model_config import model_from_json
#from tensorflow.python.keras.models import model_from_json

app = Flask(__name__)

disease_prediction = pickle.load(open('smote_xgboost_model.pkl','rb'))

#--------------------heart disease--------------------
json_file1 = open('model.json','r')
loaded_model_json1 = json_file1.read()
json_file1.close()
loaded_model = model_from_json(loaded_model_json1)
loaded_model.load_weights("model.h5")
#----------------------------end-------------------------

#--------------------sentimental analysis--------------------
json_file2 = open('sentiment_analysis_model.json','r')
loaded_model_json2 = json_file2.read()
json_file2.close()
sentimental_model = model_from_json(loaded_model_json2)
sentimental_model.load_weights("sentiment_analysis_model.h5")
#-----------------------------end----------------------------

#-----------------------------------------------RECOMMENTATION-----------------------------------------------#
'''def recommendation(age, disease, gender):
    #medi_data = pd.read_csv("./webmd.csv")
    medi_data = medi_data[~medi_data.isin([' '])]  # remove missing
    medi_data = medi_data.dropna(axis=0)  # drop null values
    label_encoder = preprocessing.LabelEncoder()

    # Encode labels in column 'Sex'.
    medi_data['Sex'] = label_encoder.fit_transform(medi_data['Sex'])

    medi_data['Condition'] = medi_data['Condition'].str.lower()
    medi_data['Reviews'] = medi_data['Reviews'].str.lower()

    medi_data.loc[(medi_data['Satisfaction'] >= 3), 'Review_Sentiment'] = 1
    medi_data.loc[(medi_data['Satisfaction'] < 3), 'Review_Sentiment'] = 0

    # to get  CV and tfID vectors for review analysis
    def tfIDfV(review):
        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(review)
        return X

    data = tfIDfV(medi_data["Reviews"])
    data.sort_indices()
    labels = to_categorical(medi_data["Review_Sentiment"], num_classes=2)
    ls = []
    preds = sentimental_model.predict(data)
    preds = np.argmax(preds, axis=1)
    for i in preds:
        if i == 1:
            ls.append("positive")
        else:
            ls.append("negative")
    medi_data['Comments'] = ls
    medi_data = medi_data[medi_data["Comments"] == "positive"]

    disease = disease.lower()
    gender = gender.lower()
    if gender == "female" or gender == "f":
        gender = 0
    else:
        gender = 1
    filtered_users = []
    # similar disease users
    for i in range(len(medi_data)):
        if disease in medi_data.iloc[i]["Condition"]:
            filtered_users.append(medi_data.iloc[i])
    filtered_user_data = pd.DataFrame(filtered_users)
    # similar age and gender users
    print(filtered_user_data)
    filtered_users = []
    for i in range(len(filtered_user_data)):
        ag = filtered_user_data.iloc[i]['Age']
        if "or" in ag:
            ag = ag.split(" ")
            ag1 = int(ag[0])
            ag2 = 100
        elif "-" in ag:
            ag = ag.split("-")
            if len(ag[-1]) > 3:
                ag1 = 0
                ag2 = 1
            else:
                ag1 = int(ag[0])
                ag2 = int(ag[1])
        if age >= ag1 and age <= ag2:
            if filtered_user_data.iloc[i]['Sex'] == gender:
                filtered_users.append(filtered_user_data.iloc[i])

    filtered_userframe = pd.DataFrame(filtered_users)

    from textblob import TextBlob

    reviews = filtered_userframe["Reviews"]

    Predict_Sentiment = []
    for review in (reviews):
        blob = TextBlob(review)
        Predict_Sentiment += [blob.sentiment.polarity]
    filtered_userframe["Predict_Sentiment"] = Predict_Sentiment
    filtered_userframe1 = filtered_userframe[filtered_userframe["Predict_Sentiment"] > 0]
    drug = {}
    for i in range(len(filtered_userframe1)):
        x = filtered_userframe1.iloc[i]
        if x['DrugId'] in drug.keys():
            drug[x['DrugId']][2].append(
                (x['EaseofUse'] + x['Effectiveness'] + x['Satisfaction'] + x['UsefulCount']) / 4)
        else:
            drug[x['DrugId']] = [x['DrugId'], x['Drug'],
                                 [(x['EaseofUse'] + x['Effectiveness'] + x['Satisfaction'] + x['UsefulCount']) / 4]]
    for i in drug.keys():
        drug[i][2] = sum(drug[i][2]) / len(drug[i][2])
        drug[i][2] = round(drug[i][2], 2)
    drug_data = pd.DataFrame(drug.values())
    drug_data.columns = ['DrugID', 'Drug_name', 'Mean_Rating']
    drug_data = drug_data.sort_values(by=['Mean_Rating'], ascending=False)
    return drug_data
'''
#------------------------------------------------FUNCTION END------------------------------------------------#

@app.route('/')
def home():
    return "hello world"

@app.route('/predict',methods = ["POST"])
def predict_dis():
    q = request.form.get('symptoms').split(',')
    age = int(request.form.get('age'))
    gender = request.form.get('gender')
    sex = int(request.form.get('sex'))
    cp = int(request.form.get('cp'))
    trestbps = int(request.form.get('trestbps'))
    fbs = int(request.form.get('fbs'))
    thalach = int(request.form.get('thalach'))
    exang =int(request.form.get('exang'))
    oldpeak = float(request.form.get('oldpeak'))
    slope = int(request.form.get('slope'))
    ca = int(request.form.get('ca'))
    thal = int(request.form.get('thal'))
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
    input_hd = np.array([age,sex,cp,trestbps,fbs,thalach,exang,oldpeak,slope,ca,thal])
    result = disease_prediction.predict(input_query)
    heart_d = loaded_model.predict(input_hd)
    for j in range(len(data1)):
        if result[0] == int(data1.iloc[j]["Disease_Labels"]):
            disease = data1.iloc[j]["Disease"]
            break
    m_v = max(heart_d[0])
    for k in range(5):
        if m_v == heart_d[0][k]:
            if k >= 2:
                disease= disease+(" and Chronic Heart Failure")
            break
    '''if len(disease)>1:
            drug1 = recommendation(int(age),str(disease[0]), str(gender))
            drug2 = recommendation(int(age),str(disease[1]), str(gender))
            temp = drug1 + drug2
    else:
        temp = recommendation(age , str(disease[0]),gender)'''
    return jsonify({'disease':str(disease)})

if __name__ == '__main__':
    app.run(debug=True)