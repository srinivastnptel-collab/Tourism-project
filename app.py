
import streamlit as st
import pandas as pd
import joblib
from huggingface_hub import hf_hub_download

MODEL_REPO = "Srinivas1969/Tourism-Model"
MODEL_FILE = "rf_model.joblib"

@st.cache_resource
def load_model():
    path = hf_hub_download(repo_id=MODEL_REPO, filename=MODEL_FILE, repo_type="model")
    return joblib.load(path)

model = load_model()

st.title("Wellness Tourism Package Prediction")

st.write("Enter customer details:")

# NOTE: Use encoded columns
# For simplicity, we accept numeric inputs only
# (You can expand this to full categorical UI later)

inputs = {}
for col in ['Unnamed: 0', 'Age', 'CityTier', 'DurationOfPitch', 'NumberOfPersonVisiting', 'NumberOfFollowups', 'PreferredPropertyStar', 'NumberOfTrips', 'Passport', 'PitchSatisfactionScore', 'OwnCar', 'NumberOfChildrenVisiting', 'MonthlyIncome', 'TypeofContact_Self Enquiry', 'Occupation_Large Business', 'Occupation_Salaried', 'Occupation_Small Business', 'Gender_Female', 'Gender_Male', 'ProductPitched_Deluxe', 'ProductPitched_King', 'ProductPitched_Standard', 'ProductPitched_Super Deluxe', 'MaritalStatus_Married', 'MaritalStatus_Single', 'MaritalStatus_Unmarried', 'Designation_Executive', 'Designation_Manager', 'Designation_Senior Manager', 'Designation_VP']:
    inputs[col] = st.number_input(col, value=0)

df = pd.DataFrame([inputs])

if st.button("Predict"):
    pred = model.predict(df)[0]
    st.success("Will Purchase" if pred==1 else "Will Not Purchase")
