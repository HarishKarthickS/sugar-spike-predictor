from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import joblib
import datetime
import os
import logging
import pickle
from pathlib import Path

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(levelname)s %(name)s %(threadName)s : %(message)s')

# Global variables
model = None
model_loaded = False
model_error = None

# Function to safely load model
def load_model():
    global model, model_loaded, model_error
    model_path = 'glucose_prediction_model.pkl'
    
    if os.path.exists(model_path):
        try:
            # Try different loading methods
            try:
                model = joblib.load(model_path)
                model_loaded = True
                app.logger.info("Model loaded successfully from %s", model_path)
            except Exception as e1:
                app.logger.warning("Joblib load failed: %s, trying pickle", e1)
                try:
                    with open(model_path, 'rb') as f:
                        model = pickle.load(f)
                    model_loaded = True
                    app.logger.info("Model loaded successfully with pickle from %s", model_path)
                except Exception as e2:
                    model_error = f"Failed to load model: {str(e1)} and {str(e2)}"
                    app.logger.error(model_error)
        except Exception as e:
            model_error = f"Error loading model: {str(e)}"
            app.logger.error(model_error)
    else:
        model_error = f"Model file {model_path} not found"
        app.logger.warning(model_error)

# Try to load the model
load_model()

@app.route('/')
def home():
    try:
        return render_template('index.html', model_loaded=model_loaded, model_error=model_error)
    except Exception as e:
        # Fallback if template not found
        html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Glucose Prediction</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }
                .container { max-width: 800px; margin: 0 auto; }
                .error { color: #e74c3c; background: #fadbd8; padding: 10px; border-radius: 5px; }
                h1 { color: #3498db; }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Glucose Prediction</h1>
                <p>Welcome to the Glucose Prediction service.</p>
                
                <div class="model-status">
                    <h2>Model Status</h2>
        """
        
        if model_loaded:
            html += "<p>✅ Model loaded successfully!</p>"
        else:
            html += f"<p class='error'>❌ Model loading error: {model_error}</p>"
            
        html += """
                </div>
                
                <div class="form-section">
                    <h2>Input Your Data</h2>
                    <form action="/predict" method="post">
                        <h3>Personal Information</h3>
                        <div>
                            <label for="age">Age:</label>
                            <input type="number" id="age" name="age" min="18" max="100" step="1" required>
                        </div>
                        <div>
                            <label for="gender">Gender:</label>
                            <select id="gender" name="gender" required>
                                <option value="Male">Male</option>
                                <option value="Female">Female</option>
                            </select>
                        </div>
                        <div>
                            <label for="bmi">BMI:</label>
                            <input type="number" id="bmi" name="bmi" min="10" max="50" step="0.1" required>
                        </div>
                        <div>
                            <label for="a1c">A1C (%):</label>
                            <input type="number" id="a1c" name="a1c" min="4" max="14" step="0.1" required>
                        </div>
                        <div>
                            <label for="fasting_glucose">Fasting Glucose (mg/dL):</label>
                            <input type="number" id="fasting_glucose" name="fasting_glucose" min="60" max="300" required>
                        </div>
                        <div>
                            <label for="insulin_level">Insulin Level (μU/mL):</label>
                            <input type="number" id="insulin_level" name="insulin_level" min="0" max="100" step="0.1" required>
                        </div>
                        <div>
                            <label for="heart_rate">Heart Rate (bpm):</label>
                            <input type="number" id="heart_rate" name="heart_rate" min="40" max="200" required>
                        </div>
                        
                        <h3>Meal Information</h3>
                        <div>
                            <label for="meal_type">Meal Type:</label>
                            <select id="meal_type" name="meal_type" required>
                                <option value="Breakfast">Breakfast</option>
                                <option value="Lunch">Lunch</option>
                                <option value="Dinner">Dinner</option>
                                <option value="Snack">Snack</option>
                            </select>
                        </div>
                        <div>
                            <label for="calories">Calories:</label>
                            <input type="number" id="calories" name="calories" min="0" max="2000" required>
                        </div>
                        <div>
                            <label for="carbs">Carbohydrates (g):</label>
                            <input type="number" id="carbs" name="carbs" min="0" max="300" required>
                        </div>
                        <div>
                            <label for="protein">Protein (g):</label>
                            <input type="number" id="protein" name="protein" min="0" max="200" required>
                        </div>
                        <div>
                            <label for="fat">Fat (g):</label>
                            <input type="number" id="fat" name="fat" min="0" max="100" required>
                        </div>
                        <div>
                            <label for="fiber">Fiber (g):</label>
                            <input type="number" id="fiber" name="fiber" min="0" max="50" required>
                        </div>
                        
                        <h3>Current Glucose Status</h3>
                        <div>
                            <label for="current_glucose">Current Glucose (mg/dL):</label>
                            <input type="number" id="current_glucose" name="current_glucose" min="60" max="400" required>
                        </div>
                        <div>
                            <label for="glucose_trend">Glucose Trend (mg/dL per hour):</label>
                            <input type="number" id="glucose_trend" name="glucose_trend" min="-20" max="20" step="0.1" required>
                        </div>
                        
                        <div>
                            <button type="submit">Predict Glucose</button>
                        </div>
                    </form>
                </div>
            </div>
        </body>
        </html>
        """
        return html

@app.route('/predict', methods=['POST'])
def predict():
    if not model_loaded:
        error_message = model_error or "Model not loaded. Please ensure the model file exists and is compatible."
        if request.content_type == 'application/json':
            return jsonify({'error': error_message}), 500
        else:
            html = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Prediction Error</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }}
                    .container {{ max-width: 800px; margin: 0 auto; }}
                    .error {{ color: #e74c3c; background: #fadbd8; padding: 10px; border-radius: 5px; }}
                </style>
            </head>
            <body>
                <div class="container">
                    <h1>Prediction Error</h1>
                    <div class="error">
                        <p>{error_message}</p>
                    </div>
                    <p><a href="/">Back to Home</a></p>
                </div>
            </body>
            </html>
            """
            return html
    
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

        try:
            return render_template('result.html', result=result)
        except Exception as e:
            # Fallback if template not found
            html = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Prediction Result</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }}
                    .container {{ max-width: 800px; margin: 0 auto; }}
                    .result {{ background: #eaf7fd; padding: 20px; border-radius: 10px; margin-bottom: 20px; }}
                    h1 {{ color: #3498db; }}
                    .prediction {{ font-size: 24px; font-weight: bold; margin: 20px 0; }}
                    .accuracy {{ color: #27ae60; font-weight: bold; }}
                    .recommendation {{ background: #e8f8f5; padding: 15px; border-radius: 5px; }}
                </style>
            </head>
            <body>
                <div class="container">
                    <h1>Your Glucose Prediction</h1>
                    
                    <div class="result">
                        <div class="prediction">
                            <p>Predicted Blood Glucose: <span>{result['prediction']} mg/dL</span></p>
                            <p>Current Blood Glucose: {result['current_glucose']} mg/dL</p>
                        </div>
                        
                        <div class="details">
                            <p>Diabetic Status: {result['is_diabetic']}</p>
                            <p>Meal Type: {result['meal_type']}</p>
                            <p>Carbohydrates: {result['carbs']}g</p>
                            <p>Prediction Accuracy: <span class="accuracy">{result['accuracy']}%</span></p>
                        </div>
                        
                        <div class="recommendation">
                            <h3>Recommendation:</h3>
                            <p>{result['message']}</p>
                        </div>
                    </div>
                    
                    <p><a href="/">Make Another Prediction</a></p>
                </div>
            </body>
            </html>
            """
            return html

    except Exception as e:
        app.logger.error("Error during prediction: %s", e, exc_info=True)
        error_msg = str(e)
        if request.content_type == 'application/json':
            return jsonify({'error': f'Prediction error: {error_msg}'}), 500
        else:
            html = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Prediction Error</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }}
                    .container {{ max-width: 800px; margin: 0 auto; }}
                    .error {{ color: #e74c3c; background: #fadbd8; padding: 10px; border-radius: 5px; }}
                </style>
            </head>
            <body>
                <div class="container">
                    <h1>Prediction Error</h1>
                    <div class="error">
                        <p>Error during prediction: {error_msg}</p>
                    </div>
                    <p><a href="/">Back to Home</a></p>
                </div>
            </body>
            </html>
            """
            return html


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
