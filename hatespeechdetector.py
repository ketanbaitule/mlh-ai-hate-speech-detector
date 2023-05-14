"""
    This File Creates the model and save DecisionTreeClassifier into "classifier.pkl" and CountVectorizer into "vectorizer.pkl"
    Use predict(text) -> [0, 1, 2] 0: Hate Speech, 1: Offensive Speech, 2: Neither of them
"""

# Import Library
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import re
import nltk
from nltk.util import pr
from nltk.corpus import stopwords
import string
import joblib
stemmer = nltk.SnowballStemmer("english")
nltk.download('stopwords')
stopword = set(stopwords.words("english"))

df = pd.read_csv('Dataset/twitter_data.csv')

# Data cleaning
def clean_text(text):
    text = str(text).lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*','', text)
    text = [word for word in text.split(' ') if word not in stopword]
    text = " ".join(text)
    return text

# Clean the dataset
df['tweet'] = df['tweet'].apply(clean_text)

x = np.array(df["tweet"])
y = np.array(df["class"])

cv = CountVectorizer()
x = cv.fit_transform(x)
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.33, random_state=42)

# Train Model
clf = DecisionTreeClassifier()
clf.fit(x_train, y_train)

# Save Model
joblib.dump(cv, 'vectorizer.pkl')
joblib.dump(clf, 'classifier.pkl')

# test_data = "I will kill you"
# result = cv.transform([test_data]).toarray()
# print(clf.predict(result))