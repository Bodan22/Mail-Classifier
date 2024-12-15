import pickle
import numpy as np
import pandas as pd
import string
from pickle import dump
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import nltk
from nltk.corpus import stopwords
import warnings
import matplotlib as plt
import seaborn as sns


warnings.simplefilter(action='ignore', category=FutureWarning)
nltk.download('stopwords')

# Load the dataset
dataset = pd.read_csv('mail classifier/emails.csv')
print(dataset) 

dataset.head() # show first 5 mails


# Check for duplicates and remove them 
dataset.drop_duplicates(inplace=True)


# Cleaning data from punctuation and stopwords and then tokenizing it into words (tokens)
def text_process(text):
    no_punct = [char for char in text if text not in string.punctuation]
    no_punct = ''.join(no_punct)
    clean = [word for word in no_punct.split() if word.lower() not in stopwords.words('english')]
    return clean

# Fit the CountVectorizer to data
message = CountVectorizer(analyzer='word').fit_transform(dataset['text'])

# Save the vectorizer
dump(message, open("mail classifier/vectorizer.pkl", "wb"))

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(message, dataset['spam'], test_size=0.20, random_state=0)

#Model creation
model = MultinomialNB()

# Model training
model.fit(X_train, y_train)

# Model saving
dump(model, open("mail classifier/model.pkl", 'wb'))

# Model predictions on test set
y_pred = model.predict(X_test)

# Model Evaluation 
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy : ", accuracy * 100)

print(classification_report(y_test, y_pred))

