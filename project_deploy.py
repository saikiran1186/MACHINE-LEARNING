import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import StratifiedKFold, train_test_split, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from category_encoders import BinaryEncoder
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report,confusion_matrix
import joblib
import streamlit as st
import gdown
import os



# Update the Google Drive file ID and model path
file_id ="1VHpIlY8J6g2mpC9HxPU2y17lP6V-1vK3"
model_path = "C:\\Users\\saikiran\\Downloads\\streamlit\\final_pl.pkl"

# Download the model from Google Drive if not already downloaded
if not os.path.exists(model_path):
    url = f'https://drive.google.com/uc?id={file_id}'
    gdown.download(url, model_path, quiet=False)

# Load the existing model
loaded_model = joblib.load(model_path)

# Title for Streamlit app
st.title("Injury Severity Prediction")

# User inputs for categorical features
acrs_report_type = st.text_input("Enter the ACRS Report Type")
collision_type = st.text_input("Enter the Collision Type")
surface_condition = st.text_input("Enter the Surface Condition")
driver_substance_abuse = st.text_input("Enter Driver Substance Abuse status")
driver_at_fault = st.text_input("Enter Driver At Fault status")
vehicle_damage_extent = st.text_input("Enter the Vehicle Damage Extent")
vehicle_body_type = st.text_input("Enter the Vehicle Body Type")
equipment_problems = st.text_input("Enter the Equipment Problems")

# User inputs for numerical features
speed_limit = st.number_input("Enter the Speed Limit", min_value=0.0)
vehicle_year = st.number_input("Enter the Vehicle Year", min_value=1990.0, max_value=2024.0)
crash_month = st.number_input("Enter the Crash Month", min_value=1, max_value=12)

if st.button("Submit"):
    # Create a dictionary of user inputs
    user_data = {
        'ACRS Report Type': acrs_report_type,
        'Collision Type': collision_type,
        'Surface Condition': surface_condition,
        'Driver Substance Abuse': driver_substance_abuse,
        'Driver At Fault': driver_at_fault,
        'Vehicle Damage Extent': vehicle_damage_extent,
        'Vehicle Body Type': vehicle_body_type,
        'Equipment Problems': equipment_problems,
        'Speed Limit': speed_limit,
        'Vehicle Year': vehicle_year,
        'Crash_Month': crash_month
    }

    # Convert user data to DataFrame
    user_df = pd.DataFrame(user_data, index=[0])
    print(user_df)

    # Preprocess user inputs and make prediction
    prediction = loaded_model.predict(user_df)
    
    st.write("The predicition is: ",{prediction[0]})