import streamlit as st
import pickle 
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import string

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = text.split()
    filtered_text = [stemmer.stem(word) for word in words if word not in stop_words]
    return ' '.join(filtered_text)

model = pickle.load(open('spam_detector.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

st.title("Welcome to Email spam classifier")
st.subheader("This is a simple email spam classifier app built using streamlit and sklearn")
st.write("Enter the email text below to check if it's spam or not")

st.sidebar.title("About")
st.sidebar.info("Made by Swayam Kumar | Powered by Streamlit and sklearn")

email_text = st.text_area("Enter the email text here:", key="email_text")
if st.button("Predict Spam"):
    if email_text.strip() == "":
        st.warning("Please enter some email text to classify.")
    else:
        email_text2 = preprocess_text(email_text)
        transformed_text = vectorizer.transform([email_text2])
        result = model.predict(transformed_text)[0]

        if result == "spam":
            st.error("The email is classified as Spam.")
        else:
            st.success("The email is classified as Not Spam.")

st.write("---")
st.caption("Project: Spam Detection | Built with Streamlit and Machine Learning")



