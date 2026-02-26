import streamlit as st
import pandas as pd
import numpy as np
import os
import csv
import re
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from difflib import get_close_matches

st.set_page_config(page_title="AI Health Companion", page_icon="🩺")

# ------------------ Load Data ------------------
BASE_DIR = os.path.dirname(__file__)

training = pd.read_csv(os.path.join(BASE_DIR, "Data", "Training.csv"))
testing = pd.read_csv(os.path.join(BASE_DIR, "Data", "Testing.csv"))

training.columns = training.columns.str.replace(r"\.\d+$", "", regex=True)
training = training.loc[:, ~training.columns.duplicated()]

cols = training.columns[:-1]
x = training[cols]
y = training['prognosis']

le = preprocessing.LabelEncoder()
y = le.fit_transform(y)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.33, random_state=42
)

model = RandomForestClassifier(n_estimators=300, random_state=42)
model.fit(x_train, y_train)

# ------------------ Dictionaries ------------------
severityDictionary = {}
description_list = {}
precautionDictionary = {}
symptoms_dict = {symptom: idx for idx, symptom in enumerate(x.columns)}

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

    # Precautions
    with open(os.path.join(BASE_DIR, "Master_data", "symptom_precaution.csv")) as csv_file:
        for row in csv.reader(csv_file):
            precautionDictionary[row[0]] = [row[1], row[2], row[3], row[4]]

load_master_data()

# ------------------ Symptom Extraction ------------------
symptom_synonyms = {
    "stomach ache": "stomach_pain",
    "belly pain": "stomach_pain",
    "tummy pain": "stomach_pain",
    "loose motion": "diarrhea",
    "high temperature": "fever",
    "temperature": "fever",
    "feaver": "fever",
    "coughing": "cough",
    "throat pain": "sore_throat",
    "shortness of breath": "breathlessness",
    "body ache": "muscle_pain",
}

def extract_symptoms(user_input):
    extracted = []
    text = user_input.lower().replace("-", " ")

    for phrase, mapped in symptom_synonyms.items():
        if phrase in text:
            extracted.append(mapped)

    for symptom in cols:
        if symptom.replace("_", " ") in text:
            extracted.append(symptom)

    words = re.findall(r"\w+", text)
    for word in words:
        close = get_close_matches(
            word,
            [s.replace("_", " ") for s in cols],
            n=1,
            cutoff=0.8,
        )
        if close:
            for sym in cols:
                if sym.replace("_", " ") == close[0]:
                    extracted.append(sym)

    return list(set(extracted))

# ------------------ Prediction ------------------
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

# ------------------ Streamlit UI ------------------

st.title("🩺 AI Health Companion Chatbot")
st.write("Describe your symptoms and get AI-based health insights.")

name = st.text_input("Your Name")
age = st.number_input("Your Age", min_value=1, max_value=120)
gender = st.selectbox("Gender", ["Male", "Female", "Other"])

symptoms_input = st.text_area(
    "Describe your symptoms (e.g., I have fever and stomach pain)"
)

if st.button("Predict Disease"):

    if not symptoms_input:
        st.error("Please describe your symptoms.")
    else:
        symptoms_list = extract_symptoms(symptoms_input)

        if not symptoms_list:
            st.error("No valid symptoms detected. Try different wording.")
        else:
            disease, confidence = predict_disease(symptoms_list)

            st.success(f"🩺 Predicted Disease: {disease}")
            st.info(f"🔎 Confidence: {confidence}%")

            st.subheader("📖 Description")
            st.write(description_list.get(disease, "No description available."))

            if disease in precautionDictionary:
                st.subheader("🛡️ Suggested Precautions")
                for p in precautionDictionary[disease]:
                    st.write("•", p)

            st.markdown("---")
            st.write("🌸 Take care of your health and consult a doctor for serious conditions.")
