from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model
with open('heart_disease_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

@app.route('/')
def home():
    return render_template('index2.html')

@app.route('/predict', methods=['POST'])
def predict():
    features = [
        float(request.form['age']),
        int(request.form['sex']),
        int(request.form['cp']),
        float(request.form['trtbps']),
        float(request.form['chol']),
        int(request.form['fbs']),
        int(request.form['restecg']),
        float(request.form['thalachh']),
        int(request.form['exng']),
        float(request.form['oldpeak']),
        int(request.form['slp']),
        int(request.form['caa']),
        int(request.form['thall'])
    ]
    
    features = np.array(features).reshape(1, -1)
    prediction = model.predict(features)[0]

    message = "You have heart disease.and take advice from your doctor some doctors names listed" if prediction == 1 else "You have no heart disease."
    return render_template('index2.html', prediction=message)

if __name__ == '__main__':
    app.run(debug=True)
