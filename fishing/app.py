from flask import Flask, render_template, url_for, request
import sqlite3
import numpy as np
import pandas as pd
from sklearn import metrics 
import warnings
import pickle
warnings.filterwarnings('ignore')
from keras.models import load_model
# from feature import 
from feature import FeatureExtraction
model = load_model('model/model.h5')


file = open("model/model.pkl","rb")
gbc = pickle.load(file)
file.close()


connection = sqlite3.connect('user_data.db')
cursor = connection.cursor()

command = """CREATE TABLE IF NOT EXISTS user(name TEXT, password TEXT, mobile TEXT, email TEXT)"""
cursor.execute(command)

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/home')
def home():
    return render_template('home.html')

@app.route('/comparison')
def comparison():
    return render_template('comparison.html')

@app.route('/userlog', methods=['GET', 'POST'])
def userlog():
    if request.method == 'POST':

        connection = sqlite3.connect('user_data.db')
        cursor = connection.cursor()

        name = request.form['name']
        password = request.form['password']

        query = "SELECT name, password FROM user WHERE name = '"+name+"' AND password= '"+password+"'"
        cursor.execute(query)

        result = cursor.fetchall()

        if result:
            return render_template('home.html')
        else:
            return render_template('index.html', msg='Sorry, Incorrect Credentials Provided,  Try Again')

    return render_template('index.html')


@app.route('/userreg', methods=['GET', 'POST'])
def userreg():
    if request.method == 'POST':

        connection = sqlite3.connect('user_data.db')
        cursor = connection.cursor()

        name = request.form['name']
        password = request.form['password']
        mobile = request.form['phone']
        email = request.form['email']
        
        print(name, mobile, email, password)

        command = """CREATE TABLE IF NOT EXISTS user(name TEXT, password TEXT, mobile TEXT, email TEXT)"""
        cursor.execute(command)

        cursor.execute("INSERT INTO user VALUES ('"+name+"', '"+password+"', '"+mobile+"', '"+email+"')")
        connection.commit()

        return render_template('index.html', msg='Successfully Registered')
    
    return render_template('index.html')

@app.route('/ML', methods=['GET', 'POST'])
def ML():
    if request.method == 'POST':
        url = request.form['Link']
        print(url)
        obj = FeatureExtraction(url)
        x = np.array(obj.getFeaturesList()).reshape(1,30) 

        y_pred =gbc.predict(x)[0]
        #1 is safe       
        #-1 is unsafe
        y_pro_phishing = gbc.predict_proba(x)[0,0]
        y_pro_non_phishing = gbc.predict_proba(x)[0,1]
        # if(y_pred ==1 ):
        pred = "It is {0:.2f} % safe to Use  ".format(y_pro_phishing*100)
        print(f"\n\n{pred}\n\n")
        return render_template('ml.html',xx =round(y_pro_non_phishing,2) , res =round(y_pro_non_phishing,2),url=url, pred=pred )
    return render_template("ml.html", xx =-1, msg="You are in ML page")


@app.route('/ANN', methods=['GET', 'POST'])
def ANN():
    if request.method == 'POST':
        Link = request.form['Link']
        print(Link)
        def preprocess_url(url):
            feature_extractor = FeatureExtraction(url)
            features = feature_extractor.getFeaturesList()
            return np.array(features).reshape(1, -1)

        # Function to predict whether the URL is phishing or not
        def predict_phishing(url):
            # Preprocess the URL and extract features
            features = preprocess_url(url)
            
            # Use the trained model to make predictions
            prediction = model.predict(features)
            
            # Get class probabilities
            probability_non_phishing = prediction[0][0]
            probability_phishing = 1 - probability_non_phishing
            
            # Round the prediction if it's a binary classification problem
            prediction_binary = np.round(prediction)
            
            return prediction_binary, probability_phishing, probability_non_phishing

        # Example URL
        # url = input("Enter the URL: ")

        # Predict whether the URL is phishing or not
        prediction, probability_phishing, probability_non_phishing = predict_phishing(Link)
        # Convert probabilities to percentages
        probability_phishing_percentage = probability_phishing * 100

        probability_non_phishing_percentage = probability_non_phishing * 100

        if prediction == 1:
            res=f"The URL is {probability_non_phishing_percentage:.2f} % SAFE "
        else:
            res=f"The URL is {probability_phishing_percentage:.2f} % NOT SAFE"
        return render_template('ann.html', res=res, url=Link)
    return render_template('ann.html', msg="You are in ANN page")

@app.route('/logout')
def logout():
    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
