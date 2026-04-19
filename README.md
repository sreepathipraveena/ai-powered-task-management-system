# 🚀 AI-Powered Task Management System

A modern **Machine Learning + NLP-based Task Management Dashboard** that automatically predicts task priority (High, Medium, Low) and provides an interactive interface for managing tasks.

---
## 🌐 Live Demo

👉 https://ai-powered-task-management-system-project.streamlit.app/
This is a deployed AI-powered task management system that predicts task priority using Machine Learning and NLP.

The app allows users to input task descriptions and get real-time AI-based priority predictions (High, Medium, Low) through an interactive dashboard.
## 📌 Project Overview

This project uses **Natural Language Processing (NLP)** and **Machine Learning** to classify task descriptions into priority levels.

It also includes a **Streamlit dashboard** for:

* Task prediction
* Task management
* Analytics visualization

---

## 🧠 Features

* 🤖 **AI-Based Priority Prediction**
* 📊 **Interactive Dashboard**
* 📝 **Task Management System**
* 📈 **Analytics & Visualization**
* 🌗 **Dark & Light Theme Support**
* 📂 **Custom Dataset Support (jiradataset.csv)**
* ⚡ **Real-time Predictions**

---

## 🛠️ Tech Stack

* Python
* Pandas, NumPy
* Scikit-learn (Naive Bayes, SVM, Logistic Regression)
* NLP (NLTK, TF-IDF)
* Streamlit (UI Dashboard)
* Matplotlib & Seaborn (Visualization)

---

## 📁 Project Structure

```
ai-task-manager/
├── data/                    # Dataset files
├── models/                  # Trained models (.pkl)
├── notebooks/               # EDA & experiments
├── .streamlit/              # UI configuration
│   └── config.toml
├── app.py                   # Streamlit dashboard
├── train.py                 # Model training script
├── utils.py                 # Preprocessing functions
├── requirements.txt         # Dependencies
└── README.md
```

---

## ⚙️ Installation & Setup

### 1️⃣ Clone Repository

```
git clone https://github.com/your-username/ai-task-manager.git
cd ai-task-manager
```

---

### 2️⃣ Create Virtual Environment

```
py -3.10 -m venv venv
venv\Scripts\activate
```

---

### 3️⃣ Install Dependencies

```
pip install -r requirements.txt
```

---

### 4️⃣ Download NLTK Data

```
python
```

```python
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
exit()
```

---

### 5️⃣ Train Model

```
python train.py
```

---

### 6️⃣ Run Application

```
streamlit run app.py
```

---

## 🌐 Application Preview

* Input a task description
* Get predicted priority instantly
* View tasks in dashboard
* Analyze task distribution

---

## 📊 Sample Inputs

| Task Description             | Expected Priority |
| ---------------------------- | ----------------- |
| Fix server crash immediately | High              |
| Update documentation         | Medium            |
| Clean workspace              | Low               |

---

## 📈 Model Performance

* Models Used:

  * Naive Bayes
  * Support Vector Machine
  * Logistic Regression

* Best Accuracy: ~62%
  *(Due to small dataset size)*

---

## 🎯 Future Enhancements

* 📅 Task deadlines & reminders
* 🗄️ Database integration (SQLite)
* 🌍 Deployment (Streamlit Cloud)
* 🔐 User authentication system
* 📊 Advanced analytics dashboard

---

## 👩‍💻 Author

**Your Name:Sreepathi Praveena,Abir Akhuli,Sumit sharma

## 📜 License

This project is for educational purpose

---

## ⭐ If you like this project

Give it a ⭐ on GitHub
