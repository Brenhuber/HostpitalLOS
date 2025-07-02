import streamlit as st
import pandas as pd 
import plotly.express as px
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

# ------------ Page Styling ------------

st.set_page_config(page_title="Hospital Stay Predictor", layout="wide")

st.markdown("""
    <style>
    .stApp {
        background-color: #1e1e2f;
        color: #f0f0f0;
    }
    .css-1cpxqw2, .css-1d391kg {
        background-color: #2b2b3c !important;
        color: #f0f0f0 !important;
        border-radius: 12px;
        padding: 2rem;
        box-shadow: 0 4px 12px rgba(0,0,0,0.3);
    }
    h1, h2, h3, h4 {
        color: #ffffff;
    }
    .stSidebar {
        background-color: #23232e;
    }
    </style>
""", unsafe_allow_html=True)


# ------------ Load & Preprocess Data ------------

df = pd.read_csv('LengthOfStay.csv')
df.drop(columns=['eid', 'vdate', 'discharged', 'facid'], inplace=True)
df = df.rename(columns={
    'dialysisrenalendstage': 'Renal_End_Stage',
    'asthma': 'Asthma',
    'irondef': 'Iron_Deficiency',
    'pneum': 'Pneumonia',
    'substancedependence': 'Substance_Dependence',
    'psychologicaldisordermajor': 'Major_Psych_Disorder',
    'depress': 'Depression',
    'psychother': 'Psychotherapy',
    'fibrosisandother': 'Fibrosis_Other',
    'malnutrition': 'Malnutrition',
    'hemo': 'Hemoglobin',
    'hematocrit': 'Hematocrit',
    'neutrophils': 'Neutrophils',
    'sodium': 'Sodium',
    'glucose': 'Glucose',
    'bloodureanitro': 'BUN',
    'creatinine': 'Creatinine',
    'bmi': 'BMI',
    'pulse': 'Pulse',
    'respiration': 'Respiration',
    'secondarydiagnosisnonicd9': 'Secondary_Diagnosis',
    'lengthofstay': 'LOS',
    'rcount_0': 'Readmit_0',
    'rcount_1': 'Readmit_1',
    'rcount_2': 'Readmit_2',
    'rcount_3': 'Readmit_3',
    'rcount_4': 'Readmit_4',
    'rcount_5+': 'Readmit_5plus',
    'gender_F': 'Female',
    'gender_M': 'Male'
})

# ------------ Prediction Model Function ------------

def predict(customer_dict, df):
    single = pd.DataFrame([customer_dict])
    single = pd.get_dummies(single)

    X = pd.get_dummies(df.drop("LOS", axis=1))
    y = df["LOS"]

    single = single.reindex(columns=X.columns, fill_value=0)
    X = X.reindex(columns=X.columns, fill_value=0)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = XGBRegressor()
    model.fit(X_train, y_train)
    accuracy = model.score(X_test, y_test)

    fi = pd.Series(model.feature_importances_, index=X_train.columns)
    prediction = model.predict(single)[0]
    return prediction, accuracy, fi

# ------------ Sidebar UI ------------

st.sidebar.title("🩺 Patient Details")
customer = {
    "Renal_End_Stage": 1 if st.sidebar.selectbox("Dialysis Renal End Stage", ["Yes", "No"]) == "Yes" else 0,
    "Asthma": 1 if st.sidebar.selectbox("Asthma", ["Yes", "No"]) == "Yes" else 0,
    "Iron_Deficiency": 1 if st.sidebar.selectbox("Iron Deficiency", ["Yes", "No"]) == "Yes" else 0,
    "Pneumonia": 1 if st.sidebar.selectbox("Pneumonia", ["Yes", "No"]) == "Yes" else 0,
    "Substance_Dependence": 1 if st.sidebar.selectbox("Substance Dependence", ["Yes", "No"]) == "Yes" else 0,
    "Major_Psych_Disorder": 1 if st.sidebar.selectbox("Major Psychological Disorder", ["Yes", "No"]) == "Yes" else 0,
    "Depression": 1 if st.sidebar.selectbox("Depression", ["Yes", "No"]) == "Yes" else 0,
    "Psychotherapy": 1 if st.sidebar.selectbox("Psychotherapy", ["Yes", "No"]) == "Yes" else 0,
    "Fibrosis_Other": 1 if st.sidebar.selectbox("Fibrosis and Other", ["Yes", "No"]) == "Yes" else 0,
    "Malnutrition": 1 if st.sidebar.selectbox("Malnutrition", ["Yes", "No"]) == "Yes" else 0,
    "Hemoglobin": st.sidebar.number_input("Hemoglobin", min_value=0.0),
    "Hematocrit": st.sidebar.number_input("Hematocrit", min_value=0.0),
    "Neutrophils": st.sidebar.number_input("Neutrophils", min_value=0.0),
    "Sodium": st.sidebar.number_input("Sodium", min_value=0.0),
    "Glucose": st.sidebar.number_input("Glucose", min_value=0.0),
    "BUN": st.sidebar.number_input("Blood Urea Nitrogen", min_value=0.0),
    "Creatinine": st.sidebar.number_input("Creatinine", min_value=0.0),
    "BMI": st.sidebar.number_input("BMI", min_value=0.0),
    "Pulse": st.sidebar.number_input("Pulse", min_value=0.0),
    "Respiration": st.sidebar.number_input("Respiration", min_value=0.0),
    "Secondary_Diagnosis": 1 if st.sidebar.selectbox("Secondary Diagnosis (Non-ICD9)", ["Yes", "No"]) == "Yes" else 0,
    "Readmit_0": 0,
    "Readmit_1": 0,
    "Readmit_2": 0,
    "Readmit_3": 0,
    "Readmit_4": 0,
    "Readmit_5plus": 0,
    "Female": 0,
    "Male": 0,
}

rcount_choice = st.sidebar.selectbox("Readmission Count", ["0", "1", "2", "3", "4", "5+"])
customer[f"Readmit_{rcount_choice if rcount_choice != '5+' else '5plus'}"] = 1

gender_choice = st.sidebar.selectbox("Gender", ["Female", "Male"])
customer[gender_choice] = 1

# ------------ Main Page ------------

st.title("🏥 Hospital Length of Stay Predictor")

if st.sidebar.button("⌚ Predict LOS"):
    col1, col2 = st.columns(2)
    prediction, accuracy, fi = predict(customer, df)
    with col1:
        st.metric("⏱️ Predicted Length of Stay", f"{round(prediction)} days")
    with col2:
        st.metric("🎯 Model Accuracy", f"{accuracy * 100:.2f}%")

    st.subheader("📊 Feature Importance")
    if fi is not None and not fi.empty:
        fig = px.bar(
            x=fi.values,
            y=fi.index,
            orientation='h',
            labels={'x': 'Importance', 'y': 'Feature'}
        )
        fig.update_layout(
            height=800,
            margin=dict(l=200, r=50, t=50, b=50),  
            yaxis=dict(tickfont=dict(size=12))    
        )
        st.plotly_chart(fig, use_container_width=True)


