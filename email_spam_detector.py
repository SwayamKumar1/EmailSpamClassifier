#import necessary libraries
import pandas as pd 
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pickle

#download nltk stopwords if not already downloaded
nltk.download('stopwords')

#Load dataset
data = pd.read_csv('spam.csv', encoding='latin-1')
data = data.rename(columns={'v1': 'label', 'v2': 'text'})
data = data[['label', 'text']]

#Preprocess text data 
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):

    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = text.split()
    filtered_words = [
        stemmer.stem(word) 
        for word in words if word not in stop_words
    ]
    return ' '.join(filtered_words)


data['text'] = data['text'].apply(preprocess_text)

#Split data into training and testing sets
x = data['text']
y = data['label']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

#Vectorize text data 
vectorizer = CountVectorizer()
x_train_vectorized = vectorizer.fit_transform(x_train)
x_test_vectorized = vectorizer.transform(x_test)

#Train Model
model = MultinomialNB()
model.fit(x_train_vectorized, y_train) 

y_pred = model.predict(x_test_vectorized)

#Evaluate Model
print("Accuracy: ", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

#Save the model and vectorizer using pickle
with open('spam_detector.pkl', 'wb') as f:
    pickle.dump(model, f)
with open('vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)