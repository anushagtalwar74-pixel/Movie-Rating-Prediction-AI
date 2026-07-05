# 🎬 CineRate AI

A professional Machine Learning web application that predicts IMDb movie ratings using movie metadata.

---

## 🚀 Features

- 🎯 Predict IMDb Movie Ratings
- 🤖 XGBoost Machine Learning Model
- 📊 Interactive Analytics Dashboard
- 📈 Movie Insights & Visualizations
- 🎭 Genre Analysis
- 🎬 Director Analysis
- 👥 Actor Analysis
- 📉 Correlation Heatmap
- 🌐 Streamlit Web Application

---

## 🛠 Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Plotly
- Streamlit
- Joblib

---

## 📂 Project Structure

```
Movie-Rating-Prediction-AI/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── assets/
│   ├── logo.png
│   ├── movie_icon.png
│   ├── bg.jpg
│   └── favicon.ico
│
├── data/
│   ├── IMDb Movies India.csv
│   └── processed_movies.csv
│
├── models/
│   ├── movie_model.pkl
│   └── preprocessor.pkl
│
└── src/
    ├── config.py
    ├── preprocessing.py
    ├── train_model.py
    ├── predict.py
    ├── visualization.py
    └── components.py
```

---

## 📊 Machine Learning Models

The following regression models were trained:

- Linear Regression
- Random Forest Regressor
- Gradient Boosting Regressor
- XGBoost Regressor

### Best Model

**XGBoost**

Performance:

- R² Score: **0.396**
- MAE: **0.798**
- RMSE: **1.060**

---

## ▶️ Run the Project

Install dependencies:

```bash
pip install -r requirements.txt
```

Run:

```bash
python -m streamlit run app.py
```

---

## 📸 Dashboard

![CineRate AI Dashboard](screenshots/dashboard.png)
---

## 👩‍💻 Author

**Anusha G Talawar**

Data Science | Machine Learning | Python