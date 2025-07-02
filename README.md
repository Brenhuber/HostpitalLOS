<h1 align="center">🏥 Hospital Stay Predictor</h1>
<p align="center"><em>Predict patient length of stay with explainable ML and interactive dashboards</em></p>

---

### 🚀 Overview

**Hospital Stay Predictor** is a sleek, Streamlit-based web app that predicts a patient's expected hospital length of stay (LOS) using clinical features and lab values. It leverages XGBoost regression, computes error reduction against a baseline, and visualizes feature importance for transparency.

---

### ✨ Features

- Input patient demographics and lab values via sidebar widgets  
- Predict hospital stay length in real-time using XGBoost  
- Compare model MAE against a baseline to measure lift  
- Display model R² score (accuracy)  
- Visualize feature importance with an interactive Plotly bar chart  
- Modern dark-themed UI with custom CSS styling  

---

### 🛠️ Tech Stack

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)
![pandas](https://img.shields.io/badge/pandas-Data%20Handling-purple?logo=pandas)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-blue?logo=scikit-learn)
![XGBoost](https://img.shields.io/badge/XGBoost-Regressor-green?logo=xgboost)
![Streamlit](https://img.shields.io/badge/Streamlit-UI-red?logo=streamlit)
![Plotly](https://img.shields.io/badge/Plotly-Interactive%20Charts-orange?logo=plotly)

---

### ⚙️ Setup Instructions

1. **Clone the repository**
   ```bash
   git clone https://github.com/Brenhuber/HospitalLOS.git
   cd HospitalLOS
   ```
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Run the app:**
   ```bash
   streamlit run app.py
   ```


