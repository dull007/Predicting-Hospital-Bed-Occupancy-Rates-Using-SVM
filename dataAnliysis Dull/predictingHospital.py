Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Load the dataset
file_path = '/content/augmented_HRD_dataset or (1).csv'  # Path to your uploaded file
data = pd.read_csv(file_path)

# Select relevant features and target variable
features = [
    "Total_no_of_Admissions",
    "Total_no_of_new_Registrations",
    "Average_Daily_inpatients",
    "Average_hospital_stay_per_patients",
    "Total_hospital_deaths",
    "No_of_beds",
    "Death_rate_%",
]
target = "Bed_occupancy_rate_%"

# Prepare the data
X = data[features]
y = data[target]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the SVM model
svm_model = SVR(kernel="rbf", C=100, gamma=0.1, epsilon=0.01)
svm_model.fit(X_train_scaled, y_train)

# Make predictions
y_pred = svm_model.predict(X_test_scaled)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print evaluation metrics
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"R-squared (R²): {r2:.4f}")

# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
import numpy as np

# Load the dataset
file_path = '/content/augmented_HRD_dataset or (1).csv'  # Update to your dataset location
data = pd.read_csv(file_path)

# Select features and target variable
features = [
    "Total_no_of_Admissions",
    "Total_no_of_new_Registrations",
    "Average_Daily_inpatients",
    "Average_hospital_stay_per_patients",
    "Total_hospital_deaths",
    "No_of_beds",
    "Death_rate_%",
]
target = "Bed_occupancy_rate_%"

X = data[features]
y = data[target]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the SVM model
svm_model = SVR(kernel="rbf", C=100, gamma=0.1, epsilon=0.01)
svm_model.fit(X_train_scaled, y_train)

# Function for making predictions
def predict_bed_occupancy(input_data):
    """
    Predicts bed occupancy given input data.
    :param input_data: A dictionary with feature values.
    :return: Predicted bed occupancy rate.
    """
    # Convert input dictionary to DataFrame
    input_df = pd.DataFrame([input_data])
    
    # Standardize input data
    input_scaled = scaler.transform(input_df)
    
    # Predict bed occupancy rate
    prediction = svm_model.predict(input_scaled)
    
    return prediction[0]

# Example input for prediction
example_input = {
    "Total_no_of_Admissions": 120,
    "Total_no_of_new_Registrations": 50,
    "Average_Daily_inpatients": 85,
    "Average_hospital_stay_per_patients": 5,
    "Total_hospital_deaths": 3,
    "No_of_beds": 100,
    "Death_rate_%": 2.5,
}

# Predict bed occupancy rate for example input
predicted_rate = predict_bed_occupancy(example_input)
print(f"Predicted Bed Occupancy Rate: {predicted_rate:.2f}%")
