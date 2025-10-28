# 📧 Email Spam Detector  

An intelligent spam classifier built with **Python**, **NLTK**, **Scikit-Learn**, and **Streamlit**.  
Detects whether a message is spam or not using Machine Learning.

---

## 🚀 Features
- ✅ Text preprocessing: lowercasing, punctuation removal, stopword filtering, stemming  
- ✅ Bag-of-words vectorization (`CountVectorizer`)  
- ✅ Classifier: Multinomial Naive Bayes  
- ✅ Interactive Streamlit web app  

---

## 📂 Files in this Repo
| File | Description |
|------|--------------|
| `train_model.py` | Trains the model and saves `spam_detector.pkl` and `vectorizer.pkl` |
| `spam_detector.pkl` | Trained ML model |
| `vectorizer.pkl` | Preprocessing vectorizer |
| `spam.csv` | Dataset (SMS Spam Collection) |
| `app.py` | Streamlit web app |
| `requirements.txt` | List of dependencies |

---

## 📊 Dataset
Uses the SMS Spam Collection Dataset from UCI / Kaggle.

## 👨‍💻 Author
Swayam Kumar
Data Science & ML Student

---

## ⚙️ How to Run Locally

```bash
# Clone the repo
git clone https://github.com/SwayamKumar1/EmailSpamClassifier.git
cd EmailSpamClassifier

# (Optional) Create a virtual environment
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
Once it runs, open the link (usually http://localhost:8501) to test your spam detector 🎯


## 🧠 Model Training
To retrain the model:

bash
python train_model.py
It will regenerate spam_detector.pkl and vectorizer.pkl.

