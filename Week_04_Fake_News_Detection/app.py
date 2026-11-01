"""
Day 25: Streamlit Web App & Deployment
Project: Fake News Detection System
This script creates an interactive web interface using Streamlit
to detect whether a news article is Real or Fake in real-time.
"""

import streamlit as st          # building the interactive web application interface
import joblib                   # loading the saved machine learning model and vectorizer

# configure the streamlit page layout and title
st.set_page_config(
    page_title="Fake News Detection System",
    page_icon="📰",
    layout="centered"
)

# display main header on the web app interface
st.title("📰 Fake News Detection System")
st.write("Enter a news article text below to check whether it is **Real** or **Fake**.")

# load the pre-trained machine learning model and TF-IDF vectorizer
# we use @st.cache_resource so the model loads only once, making the app much faster
@st.cache_resource
def load_saved_assets():
    load_model = joblib.load('fake_news_model.pkl')
    loaded_vectorizer = joblib.load('tfidf_vectorizer.pkl')
    return load_model, loaded_vectorizer

# call the function to store model and vectorizer in variables
model, vectorizer = load_saved_assets()

# create a text area widget for user to input or paste news article text
news_input = st.text_area(
    "Paste News Article Text Here:",
    placeholder="Type or paste news article content here...",
    height=150
)

# creat a prediction button 
if st.button("Check News"):
    if news_input.strip == "":
        st.warning("Please enter some news text before checking.")
    else:                                         # process the input text and make prediction
        # transform the input text using our saved TF-IDF vectorizer
        input_vectorized = vectorizer.transform([news_input])
        
        # predict using the saved SVM model
        prediction = model.predict(input_vectorized)
        
        # display the result on the web app UI
        st.subheader("Prediction Result:")
        
        # since our dataset labels were binary (e.g. 1 for real/true, 0 for fake or vice versa)
        # let's check how your model outputs the label and display it clearly
        if prediction[0] == 1:
            st.success("🟢 This news is likely **REAL**.")
        else:
            st.error("🔴 This news is likely **FAKE**.")
        

