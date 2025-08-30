# 🩺 Multiple Disease Prediction Streamlit App

This repository contains a **Streamlit-based web application** that predicts the likelihood of multiple diseases — **Diabetes, Heart Disease, and Parkinson’s Disease** — using trained machine learning models.  

The repository includes:  
- `app.py` → Streamlit app source code  
- Training notebooks for each disease model (`.ipynb`)  
- Datasets used for training and evaluation  
- `requirements.txt` → Required Python dependencies  

---

## 🚀 Features
- Predicts **Diabetes, Heart Disease, and Parkinson’s Disease**  
- Built with **Streamlit** for an interactive UI  
- Machine learning models trained on medical datasets  
- Easy to run locally or deploy on cloud platforms (Heroku, Streamlit Cloud, etc.)  

---

## 📂 Project Structure
```
.
├── app.py # Streamlit app
├── requirements.txt # Dependencies
├── Multiple disease prediction system - diabetes.ipynb
├── Multiple disease prediction system - heart.ipynb
├── Multiple disease prediction system - Parkinsons.ipynb
├── datasets/ # (Place datasets here if not included)
└── README.md

```

yaml
Copy code

---

## ⚙️ Installation & Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/multiple-disease-prediction-streamlit-app.git
   cd multiple-disease-prediction-streamlit-app
Create virtual environment (recommended)

bash
Copy code
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows
Install dependencies

bash
Copy code
pip install -r requirements.txt
Run the app

bash
Copy code
streamlit run app.py
📒 Notebooks
The repository provides training notebooks for each disease:

Diabetes Prediction → Multiple disease prediction system - diabetes.ipynb

Heart Disease Prediction → Multiple disease prediction system - heart.ipynb

Parkinson’s Disease Prediction → Multiple disease prediction system - Parkinsons.ipynb

👉 You may need to install additional libraries (e.g., scikit-learn, numpy, pandas, matplotlib) when running notebooks.

🧠 Models Used
Diabetes → Logistic Regression

Heart Disease → Logistic Regression

Parkinson’s Disease → Support Vector Machine (SVM)
(Models can be retrained via the provided notebooks.)

🎯 Future Enhancements
Add more diseases

Improve accuracy with advanced ML/DL models

Deploy on Streamlit Cloud / Hugging Face Spaces

