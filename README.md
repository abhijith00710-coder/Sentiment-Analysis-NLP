# Twitter Sentiment Analysis using NLP

## Project Overview
This project performs sentiment analysis on Twitter text data using Natural Language Processing (NLP) techniques and Machine Learning.

The application predicts whether a tweet sentiment is:

- Positive 😊
- Neutral 😐
- Negative 😔

The project is deployed using Streamlit to provide an interactive web application interface.

---

## Features

✔ Text preprocessing  
✔ Removal of special characters  
✔ Stopword removal  
✔ Lemmatization  
✔ TF-IDF feature extraction  
✔ Machine Learning classification  
✔ Interactive Streamlit interface  

---

## Technologies Used

- Python
- Pandas
- NLTK
- Scikit-learn
- Streamlit

---

## Machine Learning Model

Algorithm used:

**Multinomial Naive Bayes**

Text vectorization:

**TF-IDF Vectorizer**

---

## Project Structure

```text
Twitter-Sentiment-Analysis/
│
├── streamlit_app.py
├── Tweets.csv
├── requirements.txt
├── README.md
└── NLP_project.ipynb
```

## Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/Sentiment-Analysis-NLP.git
```

Move into project folder:

```bash
cd Sentiment-Analysis-NLP
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run Streamlit app:

```bash
streamlit run streamlit_app.py
```

---

## Dataset

Dataset used:

Twitter Tweets Dataset

---

## Future Improvements

- Improve prediction accuracy
- Deploy trained model using pickle (.pkl)
- Add visualization dashboard
- Compare multiple ML algorithms

---

## Author

Abhijith A B

Postgraduate in Statistics | Aspiring Data Scientist
