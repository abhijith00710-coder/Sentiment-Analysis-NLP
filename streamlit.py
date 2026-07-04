import streamlit as st
import pandas as pd
import nltk
import re

from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# Download required packages
nltk.download('wordnet')
nltk.download('stopwords')

# Load dataset
df=pd.read_csv("Tweets.csv")

# Remove missing values
df.dropna(inplace=True)

# Drop unwanted columns
df.drop(["textID","selected_text"],axis=1,inplace=True)

tweets=df["text"]

# Remove special characters
tweets=tweets.str.replace("[^a-zA-Z0-9 ]","",regex=True)

# Lemmatization
wn=WordNetLemmatizer()

tweets=tweets.apply(
    lambda x:" ".join(
        [wn.lemmatize(i.lower(),pos="v")
         for i in x.split()]
    )
)

# Stopwords removal
sw=stopwords.words("english")

tweets=tweets.apply(
    lambda x:" ".join(
        [i for i in x.split()
         if i not in sw]
    )
)

# Label encoding
df["sentiment"]=df["sentiment"].map(
    {
        "positive":1,
        "negative":-1,
        "neutral":0
    }
)

# TF-IDF
tfidf=TfidfVectorizer()

x=tfidf.fit_transform(tweets)
y=df["sentiment"]

# Model
model=MultinomialNB()
model.fit(x,y)

# Streamlit UI
st.title("Twitter Sentiment Analysis")

st.write(
    "Enter a sentence and predict sentiment"
)

text=st.text_input(
    "Enter text"
)

if st.button("Predict"):

    text_clean=re.sub(
        "[^a-zA-Z0-9 ]",
        "",
        text
    )

    text_clean=" ".join(
        [
            wn.lemmatize(
                word.lower(),
                pos="v"
            )
            for word in text_clean.split()
            if word.lower() not in sw
        ]
    )

    text_vector=tfidf.transform(
        [text_clean]
    )

    prediction=model.predict(
        text_vector
    )

    if prediction[0]==1:
        st.success(
            "Positive 😊"
        )

    elif prediction[0]==-1:
        st.error(
            "Negative 😔"
        )

    else:
        st.warning(
            "Neutral 😐"
        )