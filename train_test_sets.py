import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from nltk.corpus import stopwords
import os

train_data = pd.read_csv('/Users/monte/Downloads/moved_imdb_reviews_small_lemm_train.tsv', sep='\t')
test_data = pd.read_csv('/Users/monte/Downloads/moved_imdb_reviews_small_lemm_test.tsv', sep='\t')

stop_words = list(set(stopwords.words('english')))

corpus = train_data['review_lemm']
corpus_test = test_data['review_lemm']
count_tf_idf = TfidfVectorizer(stop_words=stop_words)

tf_idf = count_tf_idf.fit_transform(corpus)
tf_idf_test = count_tf_idf.transform(corpus_test)

target = train_data['pos']

X_train, X_valid, y_train, y_valid = train_test_split(tf_idf, target, random_state=10, test_size=.25)

model = LogisticRegression()
model.fit(X_train, y_train)
y_prediction = model.predict(X_valid)
accuracy = accuracy_score(y_prediction, y_valid)
print(f"""Accuracy Score: {accuracy:.2f}""")

y_prediction_test = model.predict(tf_idf_test)

test_data['pos'] = y_prediction_test

test_data.to_csv('Prediction')
cwd = os.getcwd()
print(cwd)