# -*- coding: utf-8 -*-
"""
Created on Thu Oct  1 17:17:01 2020

@author: Muhammad_Talmeez
"""

from flask import Flask, request
import numpy as np
import pandas as pd
import pickle

app=Flask(__name__)
pickle_in=open("classifier.pkl", 'rb')
classifier=pickle.load(pickle_in)

@app.route('/')
def welcome():
    return 'welcome all'

@app.route('/predict')
def predict_note_authentication():
    variance=request.args.get("variance")
    skewness=request.args.get("skewness")
    curtosis=request.args.get("curtosis")
    entropy=request.args.get("entropy")
    prediction=classifier.predict([[variance,skewness,curtosis,entropy]])
    print(prediction)
    return "The predicted value is"+str(prediction)


@app.route('/predict_file',methods=["POST"])
def predict_note_file():
    df_test=pd.read_csv(request.files.get("file"))
    print(df_test.head())
    prediction=classifier.predict(df_test) 
    return "The predicted values for csv file is: " +str(list(prediction))

if __name__=='__main__':
    app.run()