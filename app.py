from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import joblib
import datetime
import os
import logging

app = Flask(__name__)

# Configure logging
type = logging.DEBUG
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(levelname)s %(name)s %(threadName)s : %(message)s')

# Load the trained model
model_path = 'glucose_prediction_model.pkl'
if os.path.exists(model_path):
    try:
        model = joblib.load(model_path)
        app.logger.info("Model loaded successfully from %s", model_path)
    except Exception as e:
        app.logger.error("Error loading model: %s", e)
        model = None
else:
    app.logger.warning("Model file %s not found. Predictions won't work until model is available.", model_path)
    model = None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        app.logger.error("Prediction attempted but model is None")
        return jsonify({'error': 'Model not loaded. Please ensure glucose_prediction_model.pkl exists.'}), 500
    try:
        # Get user inputs from form
        person_info = {
            'age': float(request.form.get('age', 0)),
            'gender': request.form.get('gender', 'Unknown'),
            'bmi': float(request.form.get('bmi', 0)),
            'a1c': float(request.form.get('a1c', 0)),
            'fasting_glucose': float(request.form.get('fasting_glucose', 0)),
            'insulin_level': float(request.form.get('insulin_level', 0)),
            'heart_rate': float(request.form.get('heart_rate', 70))
        }
        meal_info = {
            'meal_type': request.form.get('meal_type', 'Unknown'),
            'calories': float(request.form.get('calories', 0)),
            'carbs': float(request.form.get('carbs', 0)),
            'protein': float(request.form.get('protein', 0)),
            'fat': float(request.form.get('fat', 0)),
            'fiber': float(request.form.get('fiber', 0))
        }
        # Safe conversion of glucose values
        try:
            current_glucose = float(request.form.get('current_glucose', 0))
            glucose_trend = float(request.form.get('glucose_trend', 0))
        except (ValueError, TypeError):
            current_glucose = 100.0
            glucose_trend = 0.0
            app.logger.warning("Could not convert glucose values to float, using defaults")

        # Feature preparation
        features = {
            'age': person_info['age'],
            'gender': person_info['gender'],
            'bmi': person_info['bmi'],
            'a1c': person_info['a1c'],
            'fasting_glucose': person_info['fasting_glucose'],
            'insulin_level': person_info['insulin_level'],
            'is_diabetic': person_info['a1c'] >= 6.5,
            'meal_type': meal_info['meal_type'],
            'meal_calories': meal_info['calories'],
            'meal_carbs': meal_info['carbs'],
            'meal_protein': meal_info['protein'],
            'meal_fat': meal_info['fat'],
            'meal_fiber': meal_info['fiber'],
            'hour_of_day': datetime.datetime.now().hour,
            'glucose_at_meal': current_glucose,
            'glucose_mean_1h': current_glucose,
            'glucose_std_1h': 0,
            'glucose_min_1h': current_glucose,
            'glucose_max_1h': current_glucose,
            'glucose_slope_30m': glucose_trend,
            'hr_at_meal': person_info['heart_rate'],
            'hr_mean_1h': person_info['heart_rate'],
        }

        X_pred = pd.DataFrame([features])

        # Make prediction
        prediction = float(model.predict(X_pred)[0])
        is_diabetic_str = "Yes" if features['is_diabetic'] else "No"

        # Accuracy estimation
        base_rmse = 34.5 if features['is_diabetic'] else 21.9
        carb_factor = min(1 + (meal_info['carbs'] / 300), 1.5)
        adjusted_rmse = base_rmse * carb_factor
        max_possible_rmse = 100
        accuracy = max(85, 100 - ((adjusted_rmse / max_possible_rmse) * 100))

        result = {
            'prediction': round(prediction, 1),
            'current_glucose': round(current_glucose, 1),
            'is_diabetic': is_diabetic_str,
            'meal_type': meal_info['meal_type'],
            'carbs': meal_info['carbs'],
            'accuracy': round(accuracy),
            'message': get_recommendation(prediction, features['is_diabetic'])
        }

        return render_template('result.html', result=result)

    except Exception as e:
        app.logger.error("Error during prediction: %s", e, exc_info=True)
        return jsonify({'error': f'Prediction error: {str(e)}'}), 500


def get_recommendation(predicted_glucose, is_diabetic):
    if is_diabetic:
        if predicted_glucose > 180:
            return "Your predicted glucose is higher than recommended. Consider reducing carbohydrates in this meal or discuss insulin adjustment with your healthcare provider."
        elif predicted_glucose > 140:
            return "Your predicted glucose is somewhat elevated. Consider adding more fiber to your meal or taking a short walk after eating."
        else:
            return "Your predicted glucose is within a good range for a diabetic individual after a meal."
    else:
        if predicted_glucose > 140:
            return "Your predicted glucose is higher than typical for a non-diabetic person. Consider reducing simple carbohydrates in this meal."
        elif predicted_glucose > 120:
            return "Your predicted glucose is slightly elevated. Consider pairing carbohydrates with protein and healthy fats."
        else:
            return "Your predicted glucose is within the normal range for a non-diabetic person after a meal."

if __name__ == '__main__':
    app.run(debug=True)
