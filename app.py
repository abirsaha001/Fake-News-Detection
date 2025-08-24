import streamlit as st
import joblib


vectorizer = joblib.load("vectorizer.jb")
model = joblib.load("lr_model.jb")

st.title("FAKE NEWS DETECTOR :loudspeaker:")
st.subheader("This project identifies fake news articles by cleaning text data, converting it into numerical form with TF-IDF, and using machine learning models to classify news as real or fake.")
st.write("Enter a news Article below to check whether it is Fake or Real")
news_input = st.text_area("News Article:","")

if st.button("Check News"):
    if news_input.strip():
        transform_input = vectorizer.transform([news_input])
        prediction = model.predict(transform_input)

        if prediction[0] == 1:
            st.success("The News is Real! ")
        else:
            st.error("The News is Fake! ")
else:
    st.warning("Please enter some text to analyze. ")