import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score
import streamlit as st

# Load datasets
fake = pd.read_csv("Fake.csv")
true = pd.read_csv("True.csv")

# Add labels
fake["label"] = 0
true["label"] = 1

# Combine datasets
data = pd.concat([fake, true])

# Features and labels
x = data["text"]
y = data["label"]

# Split dataset
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

# Convert text to numbers using NLP
vectorizer = TfidfVectorizer(stop_words='english')

xv_train = vectorizer.fit_transform(x_train)
xv_test = vectorizer.transform(x_test)

# Train model
model = PassiveAggressiveClassifier(max_iter=50)
model.fit(xv_train, y_train)

# Test accuracy
predictions = model.predict(xv_test)
accuracy = accuracy_score(y_test, predictions)

# Streamlit UI
st.title("Fake News Detection System")

st.write("NLP Based Machine Learning Project")

st.write(f"Model Accuracy: {round(accuracy*100,2)}%")

news = st.text_area("Enter News Article")

if st.button("Check News"):

    news_vector = vectorizer.transform([news])

    result = model.predict(news_vector)

    if result[0] == 0:
        st.error("This News is FAKE")
    else:
        st.success("This News is REAL")