from flask import Flask, request, render_template
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Memuat model yang telah disimpan
model = joblib.load('random_forest_model.pkl')

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html', prediction=None)

@app.route('/predict', methods=['POST'])
def predict():
    # Mendapatkan data dari form
    try:
        age = float(request.form['age'])
        gender = int(request.form['gender'])
        impluse = float(request.form['impluse'])
        pressurehight = float(request.form['pressurehight'])
        pressurelow = float(request.form['pressurelow'])
        glucose = float(request.form['glucose'])
        kcm = float(request.form['kcm'])
        troponin = float(request.form['troponin'])
    except ValueError:
        return render_template('index.html', prediction="Invalid input. Please enter valid numbers.")

    # Membuat array numpy dari data input
    input_features = np.array([[age, gender, impluse, pressurehight, pressurelow, glucose, kcm, troponin]])

    # Melakukan prediksi
    prediction = model.predict(input_features)

    # Menentukan hasil prediksi
    if prediction[0] == 1:
        result = 'The patient is likely to have heart disease.'
    else:
        result = 'The patient is not likely to have heart disease.'

    probabilities = model.predict_proba(input_features)
    probabilities = probabilities*100
    prob = f"Probability of having heart disease: {probabilities[0][1]} %"
    return render_template('index.html', prediction=result, probability=prob)

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=False)
