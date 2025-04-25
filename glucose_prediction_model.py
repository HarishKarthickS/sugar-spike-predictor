import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GroupKFold
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
import joblib
import os
import datetime
import warnings
warnings.filterwarnings('ignore')

# Define constants
PREDICTION_HORIZON_MINUTES = 120  # Predict glucose 2 hours after meal
GLUCOSE_HISTORY_HOURS = 2  # Use 2 hours of glucose history before meal as features
MIN_GLUCOSE_READINGS = 10  # Minimum number of glucose readings required before a meal
RANDOM_SEED = 42

print("Loading merged CGM and bio data...")
try:
    # Load the merged data
    data = pd.read_csv('merged_cgm_bio_data.csv')
    print(f"Data loaded successfully. Shape: {data.shape}")
    
    # Print column names to help debug
    print("\nAvailable columns in the dataset:")
    print(data.columns.tolist())
except Exception as e:
    print(f"Error loading data: {e}")
    exit()

# --- 1. Data Cleaning and Preprocessing ---
print("\n--- Data Cleaning and Preprocessing ---")

# Convert timestamp to datetime
data['Timestamp'] = pd.to_datetime(data['Timestamp'])

# Check for duplicate timestamps per subject
print("Checking for duplicate timestamps...")
dupes = data.duplicated(subset=['subject', 'Timestamp'], keep=False)
print(f"Found {dupes.sum()} duplicate timestamp entries")
if dupes.sum() > 0:
    # Keep the first of each duplicate
    data = data.drop_duplicates(subset=['subject', 'Timestamp'], keep='first')
    print(f"After removing duplicates, shape: {data.shape}")

# Sort by subject and timestamp
data = data.sort_values(['subject', 'Timestamp'])

# Define diabetic status based on A1c
# A1c >= 6.5% is typically considered diabetic
data['is_diabetic'] = data['A1c PDL (Lab)'] >= 6.5
print(f"Number of diabetic subjects: {data[data['is_diabetic']]['subject'].nunique()}")
print(f"Number of non-diabetic subjects: {data[~data['is_diabetic']]['subject'].nunique()}")

# Prioritize Libre GL over Dexcom GL (fill Dexcom gaps with Libre)
# This ensures we have a single consistent glucose column
data['glucose'] = data['Libre GL'].fillna(data['Dexcom GL'])
print(f"Missing values in combined glucose column: {data['glucose'].isna().sum()}")

# Calculate some basic stats
print(f"\nGlucose statistics:")
print(data.groupby('subject')['glucose'].describe().mean())

# --- 2. Identify Meal Events ---
print("\n--- Identifying Meal Events ---")

# Find rows where meal information is present (non-null Meal Type, Calories, or Carbs)
meal_mask = (
    (~data['Meal Type'].isna()) | 
    (~data['Calories'].isna()) | 
    (~data['Carbs'].isna())
)
meal_events = data[meal_mask].copy()
print(f"Found {len(meal_events)} potential meal events")

# Clean up meal events data
if len(meal_events) > 0:
    # Fill missing nutritional values with 0 (if we know there was a meal but nutritional info is incomplete)
    for col in ['Calories', 'Carbs', 'Protein', 'Fat', 'Fiber']:
        meal_events[col] = meal_events[col].fillna(0)
    
    # Create a meal identifier
    meal_events['meal_id'] = range(len(meal_events))
    
    print(f"Meal nutritional stats:")
    print(meal_events[['Calories', 'Carbs', 'Protein', 'Fat', 'Fiber']].describe())
else:
    print("No meal events found. Check the data structure.")
    exit()

# --- 3. Feature Engineering ---
print("\n--- Feature Engineering ---")

# Create function to extract features for each meal event
def extract_meal_features(meal_row, full_data):
    """
    For a given meal event, extract features from the surrounding data.
    
    Parameters:
    -----------
    meal_row : pandas Series
        A row containing meal event data
    full_data : pandas DataFrame
        The complete dataset
        
    Returns:
    --------
    pandas Series
        Features for this meal event
    """
    subject_id = meal_row['subject']
    meal_time = meal_row['Timestamp']
    
    # Subject-specific features (don't change per meal)
    subject_features = {
        'subject_id': subject_id,
        'meal_id': meal_row['meal_id'],
        'meal_timestamp': meal_time,
        'age': meal_row['Age'] if 'Age' in meal_row else 0,
        'gender': meal_row['Gender'] if 'Gender' in meal_row else 'Unknown',
        'bmi': meal_row['BMI'] if 'BMI' in meal_row else 0,
        'a1c': meal_row['A1c PDL (Lab)'] if 'A1c PDL (Lab)' in meal_row else 0,
        'fasting_glucose': meal_row['Fasting GLU - PDL (Lab)'] if 'Fasting GLU - PDL (Lab)' in meal_row else 0,
        'insulin_level': meal_row['Insulin '] if 'Insulin ' in meal_row else 0,  # Note the space after "Insulin"
        'is_diabetic': meal_row['is_diabetic'] if 'is_diabetic' in meal_row else False,
    }
    
    # Meal-specific features
    meal_features = {
        'meal_type': meal_row['Meal Type'] if not pd.isna(meal_row['Meal Type']) else 'Unknown',
        'meal_calories': meal_row['Calories'] if not pd.isna(meal_row['Calories']) else 0,
        'meal_carbs': meal_row['Carbs'] if not pd.isna(meal_row['Carbs']) else 0,
        'meal_protein': meal_row['Protein'] if not pd.isna(meal_row['Protein']) else 0,
        'meal_fat': meal_row['Fat'] if not pd.isna(meal_row['Fat']) else 0,
        'meal_fiber': meal_row['Fiber'] if not pd.isna(meal_row['Fiber']) else 0,
        'hour_of_day': meal_time.hour,
    }
    
    # Get glucose readings before the meal (for history features)
    history_start = meal_time - pd.Timedelta(hours=GLUCOSE_HISTORY_HOURS)
    subject_data = full_data[full_data['subject'] == subject_id]
    
    # Past glucose readings
    past_readings = subject_data[
        (subject_data['Timestamp'] >= history_start) & 
        (subject_data['Timestamp'] <= meal_time)
    ].copy()
    
    # Skip if we don't have enough past readings
    if len(past_readings) < MIN_GLUCOSE_READINGS:
        return None
    
    past_readings = past_readings.sort_values('Timestamp')
    
    # Calculate glucose features
    glucose_features = {
        'glucose_at_meal': past_readings.iloc[-1]['glucose'],
        'glucose_mean_1h': past_readings[past_readings['Timestamp'] >= (meal_time - pd.Timedelta(hours=1))]['glucose'].mean(),
        'glucose_std_1h': past_readings[past_readings['Timestamp'] >= (meal_time - pd.Timedelta(hours=1))]['glucose'].std(),
        'glucose_min_1h': past_readings[past_readings['Timestamp'] >= (meal_time - pd.Timedelta(hours=1))]['glucose'].min(),
        'glucose_max_1h': past_readings[past_readings['Timestamp'] >= (meal_time - pd.Timedelta(hours=1))]['glucose'].max(),
    }
    
    # Check if we can calculate slope (need at least 2 readings)
    if len(past_readings) >= 2:
        # Calculate glucose slope (rate of change) using the last 30 minutes
        last_30m = past_readings[past_readings['Timestamp'] >= (meal_time - pd.Timedelta(minutes=30))]
        if len(last_30m) >= 2:
            # Convert to minutes since first reading and calculate slope
            last_30m['minutes'] = (last_30m['Timestamp'] - last_30m['Timestamp'].iloc[0]).dt.total_seconds() / 60
            model = np.polyfit(last_30m['minutes'], last_30m['glucose'], 1)
            glucose_features['glucose_slope_30m'] = model[0]  # mg/dL per minute
        else:
            glucose_features['glucose_slope_30m'] = 0
    else:
        glucose_features['glucose_slope_30m'] = 0
    
    # Heart rate and activity features
    hr_features = {
        'hr_at_meal': past_readings.iloc[-1]['HR'] if 'HR' in past_readings.columns and not pd.isna(past_readings.iloc[-1]['HR']) else 0,
        'hr_mean_1h': past_readings[past_readings['Timestamp'] >= (meal_time - pd.Timedelta(hours=1))]['HR'].mean() if 'HR' in past_readings.columns else 0,
    }
    
    # Calculate target: Glucose level after PREDICTION_HORIZON_MINUTES
    target_time = meal_time + pd.Timedelta(minutes=PREDICTION_HORIZON_MINUTES)
    
    # Find closest reading after target_time
    future_readings = subject_data[subject_data['Timestamp'] >= target_time].copy()
    
    if len(future_readings) > 0:
        future_readings['time_diff'] = (future_readings['Timestamp'] - target_time).dt.total_seconds()
        closest_idx = future_readings['time_diff'].abs().idxmin()
        target_glucose = future_readings.loc[closest_idx, 'glucose']
        time_diff_minutes = future_readings.loc[closest_idx, 'time_diff'] / 60
        
        # Only use target if it's close enough to our desired prediction time
        if abs(time_diff_minutes) <= 15:  # Within 15 minutes of target
            target = {
                'target_glucose': target_glucose,
                'actual_prediction_minutes': time_diff_minutes + PREDICTION_HORIZON_MINUTES
            }
        else:
            return None
    else:
        return None
    
    # Combine all features
    features = {**subject_features, **meal_features, **glucose_features, **hr_features, **target}
    return pd.Series(features)

# Apply feature extraction to each meal event
print("Extracting features for each meal event...")
meal_features_list = []

for idx, meal_row in meal_events.iterrows():
    features = extract_meal_features(meal_row, data)
    if features is not None:
        meal_features_list.append(features)

if not meal_features_list:
    print("No valid meal events with sufficient data found.")
    exit()

# Create DataFrame with all meal features
meal_features_df = pd.DataFrame(meal_features_list)
print(f"Created features for {len(meal_features_df)} meal events")

# Save the feature dataset for future use
meal_features_df.to_csv('meal_features.csv', index=False)
print("Saved meal features to meal_features.csv")

# --- 4. Model Training ---
print("\n--- Model Training ---")

# Define features and target
X = meal_features_df.drop(['target_glucose', 'meal_id', 'meal_timestamp', 'subject_id', 'actual_prediction_minutes'], axis=1)
y = meal_features_df['target_glucose']

# Check for missing values in X
missing_values = X.isnull().sum()
print("\nMissing values in features:")
print(missing_values[missing_values > 0])

# Handle categorical features
categorical_features = ['gender', 'meal_type']
numerical_features = [col for col in X.columns if col not in categorical_features]

# Split data, stratifying by diabetic status and grouping by subject
# This ensures we evaluate how well the model generalizes to new subjects
groups = meal_features_df['subject_id']
X_train, X_test, y_train, y_test, groups_train, groups_test = train_test_split(
    X, y, groups, test_size=0.2, random_state=RANDOM_SEED, stratify=meal_features_df['is_diabetic']
)

print(f"Training set size: {len(X_train)}")
print(f"Testing set size: {len(X_test)}")
print(f"Number of unique subjects in training: {groups_train.nunique()}")
print(f"Number of unique subjects in testing: {groups_test.nunique()}")

# Train a gradient boosting model
print("Training gradient boosting model...")

# Create preprocessing steps with imputation
preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ]), numerical_features),
        ('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
            ('encoder', OneHotEncoder(handle_unknown='ignore'))
        ]), categorical_features)
    ],
)

# Define model pipeline
print("Using HistGradientBoostingRegressor which can handle missing values...")
model = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', HistGradientBoostingRegressor(
        max_iter=200, 
        learning_rate=0.05, 
        max_depth=4, 
        random_state=RANDOM_SEED
    ))
])

# Fit the model
model.fit(X_train, y_train)

# Make predictions
train_preds = model.predict(X_train)
test_preds = model.predict(X_test)

# Calculate metrics
train_rmse = np.sqrt(mean_squared_error(y_train, train_preds))
test_rmse = np.sqrt(mean_squared_error(y_test, test_preds))
train_mae = mean_absolute_error(y_train, train_preds)
test_mae = mean_absolute_error(y_test, test_preds)
train_r2 = r2_score(y_train, train_preds)
test_r2 = r2_score(y_test, test_preds)

print(f"\nModel Performance:")
print(f"Training RMSE: {train_rmse:.2f} mg/dL")
print(f"Testing RMSE: {test_rmse:.2f} mg/dL")
print(f"Training MAE: {train_mae:.2f} mg/dL")
print(f"Testing MAE: {test_mae:.2f} mg/dL")
print(f"Training R²: {train_r2:.3f}")
print(f"Testing R²: {test_r2:.3f}")

# --- 5. Diabetic vs Non-Diabetic Performance Analysis ---
print("\n--- Diabetic vs Non-Diabetic Performance Analysis ---")

# Get the diabetic status for the test set
test_diabetic = X_test['is_diabetic'].values

# Split predictions by diabetic status
diabetic_indices = np.where(test_diabetic)[0]
non_diabetic_indices = np.where(~test_diabetic)[0]

# Calculate metrics for each group
if len(diabetic_indices) > 0:
    diabetic_rmse = np.sqrt(mean_squared_error(y_test.iloc[diabetic_indices], test_preds[diabetic_indices]))
    diabetic_mae = mean_absolute_error(y_test.iloc[diabetic_indices], test_preds[diabetic_indices])
    print(f"Diabetic subjects (n={len(diabetic_indices)}):")
    print(f"  RMSE: {diabetic_rmse:.2f} mg/dL")
    print(f"  MAE: {diabetic_mae:.2f} mg/dL")

if len(non_diabetic_indices) > 0:
    non_diabetic_rmse = np.sqrt(mean_squared_error(y_test.iloc[non_diabetic_indices], test_preds[non_diabetic_indices]))
    non_diabetic_mae = mean_absolute_error(y_test.iloc[non_diabetic_indices], test_preds[non_diabetic_indices])
    print(f"Non-diabetic subjects (n={len(non_diabetic_indices)}):")
    print(f"  RMSE: {non_diabetic_rmse:.2f} mg/dL")
    print(f"  MAE: {non_diabetic_mae:.2f} mg/dL")

# --- 6. Save the model ---
print("\n--- Saving the Model ---")
joblib.dump(model, 'glucose_prediction_model.pkl')
print("Model saved to glucose_prediction_model.pkl")

# --- 7. Create function for personalized predictions ---
def predict_glucose_after_meal(person_info, meal_info, current_glucose, glucose_trend):
    """
    Make a personalized prediction of glucose level after a meal.
    
    Parameters:
    -----------
    person_info : dict
        Personal information including age, gender, bmi, a1c, etc.
    meal_info : dict
        Meal information including carbs, protein, fat, etc.
    current_glucose : float
        Current glucose level (mg/dL)
    glucose_trend : float
        Trend of glucose (mg/dL per minute) in the last 30 minutes
        
    Returns:
    --------
    float
        Predicted glucose level after meal
    """
    # Create feature dictionary
    features = {
        'age': person_info.get('age', 0),
        'gender': person_info.get('gender', 'Unknown'),
        'bmi': person_info.get('bmi', 0),
        'a1c': person_info.get('a1c', 0),
        'fasting_glucose': person_info.get('fasting_glucose', 0),
        'insulin_level': person_info.get('insulin_level', 0),
        'is_diabetic': person_info.get('a1c', 0) >= 6.5,
        'meal_type': meal_info.get('meal_type', 'Unknown'),
        'meal_calories': meal_info.get('calories', 0),
        'meal_carbs': meal_info.get('carbs', 0),
        'meal_protein': meal_info.get('protein', 0),
        'meal_fat': meal_info.get('fat', 0),
        'meal_fiber': meal_info.get('fiber', 0),
        'hour_of_day': datetime.datetime.now().hour,
        'glucose_at_meal': current_glucose,
        'glucose_mean_1h': current_glucose,  # Simplified
        'glucose_std_1h': 0,  # Simplified
        'glucose_min_1h': current_glucose,  # Simplified
        'glucose_max_1h': current_glucose,  # Simplified
        'glucose_slope_30m': glucose_trend,
        'hr_at_meal': person_info.get('heart_rate', 70),
        'hr_mean_1h': person_info.get('heart_rate', 70),
    }
    
    # Convert to DataFrame
    X_pred = pd.DataFrame([features])
    
    # Load the model
    loaded_model = joblib.load('glucose_prediction_model.pkl')
    
    # Make prediction
    prediction = loaded_model.predict(X_pred)[0]
    
    return prediction

# Example of how to use the prediction function
example_person = {
    'age': 45,
    'gender': 'F',
    'bmi': 28.0,
    'a1c': 6.7,  # Diabetic
    'fasting_glucose': 130,
    'insulin_level': 15.0,
    'heart_rate': 75
}

example_meal = {
    'meal_type': 'Lunch',
    'calories': 600,
    'carbs': 60,
    'protein': 25,
    'fat': 20,
    'fiber': 8
}

# Show an example prediction
print("\n--- Example Prediction ---")
predicted_glucose = predict_glucose_after_meal(
    example_person, 
    example_meal, 
    current_glucose=120, 
    glucose_trend=0.2  # Rising slightly
)
print(f"Predicted glucose after {PREDICTION_HORIZON_MINUTES} minutes: {predicted_glucose:.1f} mg/dL")

# --- 8. Plot actual vs predicted values ---
plt.figure(figsize=(10, 6))
plt.scatter(y_test, test_preds, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel('Actual Glucose (mg/dL)')
plt.ylabel('Predicted Glucose (mg/dL)')
plt.title('Actual vs Predicted Glucose Levels')
plt.savefig('prediction_performance.png')
print("Saved prediction performance plot to prediction_performance.png") 