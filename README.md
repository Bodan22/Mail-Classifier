# Email Spam Classifier

A machine learning model built with Python that classifies emails as spam or non-spam using Natural Language Processing (NLP) and the Multinomial Naive Bayes algorithm.

## Overview

This project implements an email spam classifier using scikit-learn's MultinomialNB classifier. The model processes email text data, removes stopwords and punctuation, and uses bag-of-words features to classify messages as either spam or legitimate.

## Features

- Text preprocessing with NLTK
- Vectorization using CountVectorizer
- Multinomial Naive Bayes classification
- Model persistence (saving and loading)
- Performance metrics evaluation
- Handles duplicate entries automatically

## Requirements

```
numpy
pandas
scikit-learn
nltk
matplotlib
seaborn
```


## Dataset

The project uses a dataset (`emails.csv`) containing:
- 5,728 email messages
- 2 columns: 'text' (email content) and 'spam' (binary classification)
- Balanced between spam and non-spam classes

## Project Structure

```
mail_classifier/
│
├── classifier.py          # Main classification script
├── emails.csv            # Dataset
├── model.pkl             # Trained model (generated)
├── vectorizer.pkl        # Fitted vectorizer (generated)
├── requirements.txt      # Project dependencies
└── README.md            # This file
```

## Implementation Details

### Data Preprocessing
- Removal of duplicates
- Punctuation removal
- Stopwords removal using NLTK
- Text tokenization

### Feature Engineering
- Text vectorization using CountVectorizer
- Bag-of-words representation

### Model Training
1. Data splitting (80% training, 20% testing)
2. Model training using MultinomialNB
3. Model evaluation using various metrics

## Usage

1. Ensure all requirements are installed:
```bash
pip install -r requirements.txt
```

2. Download required NLTK data:
```python
import nltk
nltk.download('stopwords')
```

3. Run the classifier:
```bash
python classifier.py
```

## Model Performance

The model achieves:
- High accuracy in distinguishing between spam and non-spam emails
- Balanced precision and recall
- Detailed metrics available in classification report


## Model Persistence

The trained model and vectorizer are saved to disk:
- `model.pkl`: Contains the trained MultinomialNB model
- `vectorizer.pkl`: Contains the fitted CountVectorizer

## Future Improvements

1. Implement cross-validation
2. Add more feature engineering techniques
3. Try different classification algorithms
4. Add a web interface
5. Implement real-time email classification
6. Add support for multiple languages

## Acknowledgments

- NLTK for natural language processing tools
- scikit-learn for machine learning implementation
- Dataset contributors
