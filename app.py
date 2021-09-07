from flask import Flask
from flask import Flask, render_template, redirect, request
import joblib
import numpy as np

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

# def get_params():
#     parameters = []
#     parameters.append(request.form['age'])
#     parameters.append(request.form['sex'])
#     parameters.append(request.form['cpain'])
#     parameters.append(request.form['restbp'])
#     parameters.append(request.form['cho'])
#     parameters.append(request.form['fbs'])
#     parameters.append(request.form['ecg'])
#     parameters.append(request.form['mrate'])
#     parameters.append(request.form['angina'])
#     parameters.append(request.form['oldpeak'])
#     parameters.append(request.form['St'])
#     parameters.append(request.form['vessels'])
#     parameters.append(request.form['hb'])

#     return parameters

@app.route('/predict', methods = ['POST'])
def predict():
    model = joblib.load('mygrid.pkl')

    if request.method == 'POST':
        params = [x for x in request.form.values()]

    input_features = np.asarray(params).reshape(1,-1)
    preds = model.predict(input_features)
    return render_template('display.html', preds = preds)


if __name__ == '__main__':
    app.run(debug=True)