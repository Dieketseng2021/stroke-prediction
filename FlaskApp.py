# -*- coding: utf-8 -*-
import numpy as np
import pickle
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)
pickle_in = open("stroke_predict_class.pkl", "rb")
classifier = pickle.load(pickle_in)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [x for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = classifier.predict(final_features)

    prediction_text = "Prediction: The patient WILL have a stroke" if prediction[0] == 1 \
        else "Prediction: The patient will NOT have a stroke"
    return render_template('index.html', prediction_text=prediction_text)


if __name__ == '__main__':
    app.run()
