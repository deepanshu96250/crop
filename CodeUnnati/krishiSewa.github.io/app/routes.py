from flask import render_template, request
import numpy as np
import pickle
from . import app

# Load the trained model and scalers
with open('best_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)
with open('standscaler.pkl', 'rb') as scaler_file:
    sc = pickle.load(scaler_file)
with open('minmaxscaler.pkl', 'rb') as minmax_file:
    ms = pickle.load(minmax_file)

# Login route
@app.route('/')
def login():
    return render_template('login.html')

# Index route
@app.route('/index')
def index():
    return render_template('index.html')


















# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    # Get user input from the form
    N = request.form['Nitrogen']
    P = request.form['Phosporus']
    K = request.form['Potassium']
    temp = request.form['Temperature']
    humidity = request.form['Humidity']
    ph = request.form['Ph']
    rainfall = request.form['Rainfall']
    
    user_input = {
        'nitrogen': N,
        'phosphorus': P,
        'potassium': K,
        'temperature': temp,
        'humidity': humidity,
        'ph': ph,
        'rainfall': rainfall
    }

    feature_list = [N, P, K, temp, humidity, ph, rainfall]
    single_pred = np.array(feature_list).reshape(1, -1)

    scaled_features = ms.transform(single_pred)
    final_features = sc.transform(scaled_features)
    prediction = model.predict(final_features)

    crop_dict = {1: "Rice", 2: "Maize", 3: "Jute", 4: "Cotton", 5: "Coconut", 6: "Papaya", 7: "Orange",
                 8: "Apple", 9: "Muskmelon", 10: "Watermelon", 11: "Grapes", 12: "Mango", 13: "Banana",
                 14: "Pomegranate", 15: "Lentil", 16: "Blackgram", 17: "Mungbean", 18: "Mothbeans",
                 19: "Pigeonpeas", 20: "Kidneybeans", 21: "Chickpea", 22: "Coffee"}

    if prediction[0] in crop_dict:
        crop = crop_dict[prediction[0]]
        result = "{}".format(crop)
    else:
        result = "Sorry, we could not determine the best crop to be cultivated with the provided data."
    return render_template('result.html', result=result, user_input=user_input)
