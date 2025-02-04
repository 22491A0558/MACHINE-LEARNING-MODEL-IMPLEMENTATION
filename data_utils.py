import pandas as pd
import numpy as np
from sklearn.datasets import fetch_20newsgroups
import re
import matplotlib.pyplot as plt
import seaborn as sns
import nltk

# Download required NLTK data
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

def load_data():
    """Load and prepare the dataset."""
    # Using 20 newsgroups dataset as a substitute for email data
    categories = ['comp.sys.mac.hardware', 'misc.forsale']
    newsgroups = fetch_20newsgroups(subset='all', categories=categories, remove=('headers', 'footers', 'quotes'))

    # Create DataFrame
    df = pd.DataFrame({
        'text': newsgroups.data,
        'label': newsgroups.target  # 0 for ham, 1 for spam
    })

    # Clean the text
    df['text'] = df['text'].apply(preprocess_text)

    return df

def preprocess_text(text):
    """Clean and preprocess text data."""
    if not isinstance(text, str):
        return ""

    # Convert to lowercase
    text = text.lower()

    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text

def create_visualizations(data):
    """Create visualization plots for the dataset."""
    figs = []

    # 1. Class distribution
    plt.figure(figsize=(10, 6))
    sns.countplot(data=data, x='label')
    plt.title('Distribution of Ham vs Spam Emails')
    plt.xlabel('Label (0: Ham, 1: Spam)')
    plt.ylabel('Count')
    figs.append(plt.gcf())
    plt.close()

    # 2. Text length distribution
    data['text_length'] = data['text'].str.len()
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=data, x='label', y='text_length')
    plt.title('Text Length Distribution by Class')
    plt.xlabel('Label (0: Ham, 1: Spam)')
    plt.ylabel('Text Length')
    figs.append(plt.gcf())
    plt.close()

    # 3. Word count distribution
    # Simplified word count using string split instead of word_tokenize
    data['word_count'] = data['text'].str.split().str.len()
    plt.figure(figsize=(10, 6))
    sns.violinplot(data=data, x='label', y='word_count')
    plt.title('Word Count Distribution by Class')
    plt.xlabel('Label (0: Ham, 1: Spam)')
    plt.ylabel('Word Count')
    figs.append(plt.gcf())
    plt.close()

    return figs