# 🩺 Diabetes Readmission Prediction

This project predicts the likelihood of hospital readmission for diabetic patients using Machine Learning, Deep Learning, and Reinforcement Learning techniques. It also includes an interactive frontend for user input and risk prediction.

---

## 📌 Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Objective](#objective)
- [Project Structure](#project-structure)
- [Tech Stack](#tech-stack)
- [Implementation Details](#implementation-details)
  - [1. Exploratory Data Analysis (EDA)](#1-exploratory-data-analysis-eda)
  - [2. Feature Engineering](#2-feature-engineering)
  - [3. Model Building](#3-model-building)
  - [4. Deep Learning](#4-deep-learning)
  - [5. Reinforcement Learning (RL)](#5-reinforcement-learning-rl)
  - [6. Frontend (UI)](#6-frontend-ui)
- [Model Deployment](#model-deployment)
- [Large File Handling](#large-file-handling)
- [How to Run](#how-to-run)
- [Future Work](#future-work)
- [License](#license)
- [Acknowledgements](#acknowledgements)

---

## 🧠 Overview

Hospital readmission of diabetic patients is a critical healthcare concern. Reducing readmissions improves patient outcomes and lowers costs. This project uses supervised and reinforcement learning techniques to predict readmission risk and optimize intervention policies.

---

## 📊 Dataset

- **Source:** [UCI Diabetes Readmission Dataset](https://archive.ics.uci.edu/ml/datasets/diabetes+130-us+hospitals+for+years+1999-2008)
- **Size:** ~100,000 records
- **Features:** 50+
- **Target:** `readmitted` (Yes/No/<30)

---

## 🎯 Objective

1. Predict whether a diabetic patient will be readmitted.
2. Optimize hospital intervention policies using Reinforcement Learning.
3. Build an interactive UI to test models with real inputs.
4. Avoid uploading large model files on GitHub and handle LFS properly.

---

## 🗂 Project Structure


DiabetesProject/
├── notebooks/
│   ├── EDA.ipynb
│   ├── Feature_Engineering.ipynb
│   ├── ML_Models.ipynb
│   ├── DL_Model.ipynb
│   ├── RL_Environment.ipynb
│
├── models/
│   └── Random_Forest.pkl  # (ignored in Git)
│
├── webapp/
│   ├── vite-react-ui/
│   │   ├── src/
│   │   └── ... (frontend files)
│   │
│   └── flask-api/
│       └── app.py
│
├── .gitignore
├── requirements.txt
└── README.md


## 🧰 Tech Stack

- **Languages:** Python, JavaScript
- **Frontend:** React + Tailwind CSS (Vite)
- **Backend:** Flask API
- **ML/DL Libraries:** scikit-learn, XGBoost, Keras/TensorFlow
- **RL:** OpenAI Gym, custom environment
- **Deployment:** GitHub, Google Colab, (optional: Hugging Face / Streamlit)

---


## ⚙️ Implementation Details

### 1. Exploratory Data Analysis (EDA)
- Missing value treatment
- Class imbalance observation
- Correlation matrix and feature importance

### 2. Feature Engineering
- Feature selection (Chi2, mutual_info)
- Encoding (OneHot, LabelEncoding)
- Normalization

### 3. Model Building
- ✅ Logistic Regression  
- ✅ Decision Tree  
- ✅ Random Forest  
- ✅ XGBoost  
- ✅ Hyperparameter tuning (GridSearchCV)

### 4. Deep Learning
- Built Neural Network using Keras
- Compared performance with classical ML models
- Used early stopping and dropout regularization

### 5. Reinforcement Learning (RL)
- Defined custom environment simulating patient-hospital interaction
- Action space: intervention types
- Reward function: based on reduction in readmission
- Used Q-Learning & DQN agents

### 6. Frontend (UI)
- Built using React + Tailwind CSS (Vite)
- Input form for user medical details
- Predict button integrated with Flask API
- Animated and responsive interface

---

## 🚀 Model Deployment

The model is served through a **Flask API**, and predictions are made through a modern **React frontend**. Deployment options include:

- Render / Railway (for Flask backend)
- Vercel / Netlify (for React frontend)
- Streamlit (for fast deployment)
- Google Colab (for interactive notebooks)

---

## 📦 Large File Handling

The file `Random_Forest.pkl` exceeds GitHub's 100MB limit and is:

- ❌ Removed from Git history using `git filter-branch`
- ✅ Added to `.gitignore`
- ✅ Stored locally and not pushed to GitHub
- ✅ Can be uploaded separately via Google Drive if needed

---

## 🛠 How to Run

### 🔹 Backend (Flask)
```bash
cd webapp/flask-api
pip install -r requirements.txt
python app.py

### 🔹 Frontend (React + Vite)

```bash
cd webapp/vite-react-ui
npm install
npm run dev

---

## 🚧 Future Work

- Deploy RL agent as a backend microservice  
- Store predictions and user data in a database (e.g., MongoDB)  
- Add model interpretability using **SHAP** or **LIME**  
- Provide downloadable reports based on prediction outcomes  
- Integrate user feedback for continuous model improvement  

---

## 📄 License

MIT License © 2025 **Jatin Thakur**

---

## 🙌 Acknowledgements

- [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/index.php)  
- Libraries: **scikit-learn**, **XGBoost**, **Keras**, **TensorFlow**  
- **OpenAI Gym** for Reinforcement Learning simulation  
- The **GitHub Community** for support on Git LFS and clean versioning practices
