import streamlit as st
import pandas as pd
import numpy as np
import os
import csv
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# ------------------ Page Config ------------------
st.set_page_config(page_title="AI Health Companion", page_icon="🩺")

# ------------------ Load Data ------------------
BASE_DIR = os.path.dirname(__file__)

training = pd.read_csv(os.path.join(BASE_DIR, "Data", "Training.csv"))
testing = pd.read_csv(os.path.join(BASE_DIR, "Data", "Testing.csv"))

# Clean duplicate column names
training.columns = training.columns.str.replace(r"\.\d+$", "", regex=True)
training = training.loc[:, ~training.columns.duplicated()]

cols = training.columns[:-1]
x = training[cols]
y = training["prognosis"]

# Encode target
le = preprocessing.LabelEncoder()
y = le.fit_transform(y)

# Train model
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.33, random_state=42
)

model = RandomForestClassifier(n_estimators=300, random_state=42)
model.fit(x_train, y_train)

# ------------------ Load Master Data ------------------
severityDictionary = {}
description_list = {}
precautionDictionary = {}

def load_master_data():
    # Severity
    with open(os.path.join(BASE_DIR, "Master_data", "symptom_severity.csv")) as csv_file:
        for row in csv.reader(csv_file):
            try:
                severityDictionary[row[0]] = int(row[1])
            except:
                pass

    # Description
    with open(os.path.join(BASE_DIR, "Master_data", "symptom_Description.csv")) as csv_file:
        for row in csv.reader(csv_file):
            description_list[row[0]] = row[1]

    # Precaution
    with open(os.path.join(BASE_DIR, "Master_data", "symptom_precaution.csv")) as csv_file:
        for row in csv.reader(csv_file):
            precautionDictionary[row[0]] = [row[1], row[2], row[3], row[4]]

load_master_data()

# Create symptom dictionary
symptoms_dict = {symptom: idx for idx, symptom in enumerate(x.columns)}

# ------------------ Prediction Function ------------------
def predict_disease(symptoms_list):
    input_vector = np.zeros(len(symptoms_dict))
    for symptom in symptoms_list:
        if symptom in symptoms_dict:
            input_vector[symptoms_dict[symptom]] = 1

    pred_proba = model.predict_proba([input_vector])[0]
    pred_class = np.argmax(pred_proba)
    disease = le.inverse_transform([pred_class])[0]
    confidence = round(pred_proba[pred_class] * 100, 2)

    return disease, confidence


# ================== STREAMLIT UI ==================

st.title("🩺 AI Health Companion Chatbot")
st.write("Select your symptoms to get AI-based health insights.")

# User Details
name = st.text_input("Your Name")
age = st.number_input("Your Age", min_value=1, max_value=120)
gender = st.selectbox("Gender", ["Male", "Female", "Other"])

st.markdown("---")

# Symptom Selection
st.subheader("Select Your Symptoms")

symptom_options = [sym.replace("_", " ") for sym in cols]

selected_readable = st.multiselect(
    "Choose symptoms:",
    symptom_options
)

# Convert back to model format
selected_symptoms = [sym.replace(" ", "_") for sym in selected_readable]

st.markdown("---")

if st.button("Predict Disease"):

    if not selected_symptoms:
        st.error("Please select at least one symptom.")
    else:
        disease, confidence = predict_disease(selected_symptoms)

        st.success(f"🩺 Predicted Disease: {disease}")
        st.info(f"🔎 Confidence: {confidence}%")

        st.progress(int(confidence))

        st.subheader("📖 Description")
        st.write(description_list.get(disease, "No description available."))

        if disease in precautionDictionary:
            st.subheader("🛡️ Suggested Precautions")
            for p in precautionDictionary[disease]:
                st.write("•", p)

        st.markdown("---")
        st.write("⚠️ This prediction is AI-based and not a medical diagnosis.")


