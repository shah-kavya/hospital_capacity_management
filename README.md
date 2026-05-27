CareFlow Command Center  
Healthcare Capacity & Surge Management Dashboard  

---

Project Overview:

This project is an interactive healthcare analytics dashboard designed to help hospital operations managers monitor patient demand, identify bottlenecks, and predict short-term surge risk.

The system combines historical data, current system status, and machine learning-based forecasting to support proactive decision-making.

---

Key Features:

- Interactive dashboard with filters (city, age group, gender, encounter type)  
- KPI monitoring (patients, encounters, length of stay)  
- 72-hour patient demand forecasting  
- Risk classification (Low / Medium / High)  
- Decision-support recommendations  

---

Technologies Used:

- Python (Flask)  
- Pandas, NumPy  
- Scikit-learn (Machine Learning)  
- Plotly (Visualization)  
- HTML, CSS  

---

How to Run the Application:

1. Install Python (version 3.9 or above)

2. Install required libraries:
`pip install -r requirements.txt`

3. Run the application:
`python app.py`

4. Open the application in browser:
`http://127.0.0.1:5000/`

---

Data Used:

- patients.csv → patient demographics  
- encounters.csv → encounter-level data  

---

Limitations:

- Uses synthetic healthcare data (Synthea)  
- Forecast model is simplified  
- Does not include real-time hospital data  

---

Team:

Kavya Shah  
Piyush Bhattarai  
Shashank Mallesh  