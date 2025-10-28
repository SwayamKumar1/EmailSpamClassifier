# 📧 EMAIL SPAM CLASSIFIER  

A **Machine Learning Web App** that detects whether an email or message is **Spam** or **Not Spam**, built using **Python**, **Scikit-Learn**, **NLTK**, and **Streamlit**.  

---

## 🔍 OVERVIEW  

This project demonstrates a complete end-to-end **text classification pipeline** — from data preprocessing and model training to deployment in an interactive web app.  

Users can type or paste a message into the Streamlit interface and instantly see the prediction result (Spam or Ham).  

---

## ⚙️ TECH STACK  

| Component | Description |
|------------|-------------|
| **Python** | Core programming language |
| **Pandas / NumPy** | Data manipulation and numerical operations |
| **NLTK** | Natural Language Processing (tokenization, stopwords, stemming) |
| **Scikit-Learn** | Machine learning model training (Naive Bayes) |
| **Streamlit** | Lightweight web app framework for model deployment |

---

## 🧠 MODEL PIPELINE  

1. **Data Cleaning** — Lowercasing, punctuation removal, stopword filtering  
2. **Text Preprocessing** — Tokenization + Porter Stemming  
3. **Feature Extraction** — CountVectorizer (Bag-of-Words)  
4. **Model** — Multinomial Naive Bayes Classifier  
5. **Prediction** — Output label as “Spam” or “Not Spam”

---

## 📂 PROJECT STRUCTURE  

A clear breakdown of the repository files and their purpose:

EmailSpamClassifier/
│
├── 📄 app.py # Streamlit web app for live spam detection
├── 📄 train_model.py # Model training script
├── 📊 spam.csv # Dataset (SMS Spam Collection)
│
├── 🧠 spam_detector.pkl # Trained Naive Bayes model
├── 🔤 vectorizer.pkl # Fitted CountVectorizer
│
├── 📦 requirements.txt # Dependencies list
└── 📝 README.md # Project documentation


---

## 🧾 DATASET  

Dataset: **SMS Spam Collection Dataset** from [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/sms+spam+collection).  
It contains 5,574 labeled SMS messages as **Spam** or **Ham (Not Spam)**.

---

## 💻 INSTALLATION & USAGE  

### 1️⃣ Clone the Repository  
```bash
git clone https://github.com/SwayamKumar1/EmailSpamClassifier.git
cd EmailSpamClassifier
2️⃣ (Optional) Create a Virtual Environment
bash
Copy code
python -m venv venv
# Activate it:
# Windows:
venv\Scripts\activate
# macOS / Linux:
source venv/bin/activate
3️⃣ Install Requirements
bash
Copy code
pip install -r requirements.txt
4️⃣ Run the Streamlit App
bash
Copy code
streamlit run app.py
Then open the provided URL (usually http://localhost:8501) in your browser.

🧩 MODEL RETRAINING
To retrain the classifier using the dataset:

bash

python train_model.py
This will regenerate:

spam_detector.pkl

vectorizer.pkl


```
🧑‍💻 AUTHOR
Swayam Kumar
Data Science & AI Student

📍 Location: India




