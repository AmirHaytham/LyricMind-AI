import pandas as pd
import numpy as np
from collections import Counter
import re
from textblob import TextBlob
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize

def extract_text_features(text):
    """Extract numerical features from text data"""
    if pd.isna(text):
        return pd.Series({
            'word_count': 0,
            'char_count': 0,
            'avg_word_length': 0,
            'sentence_count': 0,
            'unique_words': 0,
            'complexity_score': 0,
            'sentiment_polarity': 0
        })
    
    # Basic text cleaning
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', '', text)
    
    # Tokenization
    try:
        words = word_tokenize(text)
        sentences = sent_tokenize(text)
    except LookupError:
        # Fallback tokenization if NLTK resources are not available
        words = text.split()
        sentences = text.split('.')
    
    # Calculate features
    word_count = len(words)
    char_count = len(text)
    avg_word_length = char_count / word_count if word_count > 0 else 0
    sentence_count = len(sentences)
    unique_words = len(set(words))
    
    # Complexity score (ratio of unique words to total words)
    complexity_score = unique_words / word_count if word_count > 0 else 0
    
    # Sentiment analysis
    blob = TextBlob(text)
    sentiment_polarity = blob.sentiment.polarity
    
    return pd.Series({
        'word_count': word_count,
        'char_count': char_count,
        'avg_word_length': avg_word_length,
        'sentence_count': sentence_count,
        'unique_words': unique_words,
        'complexity_score': complexity_score,
        'sentiment_polarity': sentiment_polarity
    })

def add_numerical_features(df, text_column):
    """Add numerical features to the dataframe based on text analysis"""
    # Extract features for each row
    features_df = df[text_column].apply(extract_text_features)
    
    # Combine with original dataframe
    return pd.concat([df, features_df], axis=1)
