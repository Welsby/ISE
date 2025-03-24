import pandas as pd
import numpy as np
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from gensim.models import KeyedVectors

# Load Google's Pre-trained Word2Vec Model
word_vectors = KeyedVectors.load_word2vec_format("../datasets/GoogleNews-vectors-negative300.bin", binary=True)

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Define stopwords
NLTK_stop_words_list = stopwords.words('english')
custom_stop_words_list = ['...', 'br', 'http', 'https']
final_stop_words_list = NLTK_stop_words_list + custom_stop_words_list


def remove_html(text):
    return re.sub(r'<.*?>', '', text)


def remove_emoji(text):
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"
                               u"\U0001F300-\U0001F5FF"
                               u"\U0001F680-\U0001F6FF"
                               u"\U0001F1E0-\U0001F1FF"
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)


def remove_stopwords(text):
    return " ".join([word for word in text.split() if word.lower() not in final_stop_words_list])


def clean_str(text):
    """Clean text by removing unwanted characters, extra spaces, and formatting issues."""
    text = re.sub(r"[^A-Za-z0-9(),.!?\'\`]", " ", text)
    text = re.sub(r"\'s", " \'s", text)
    text = re.sub(r"\'ve", " \'ve", text)
    text = re.sub(r"\)", " ) ", text)
    text = re.sub(r"\?", " ? ", text)
    text = re.sub(r"\s{2,}", " ", text)
    text = re.sub(r"\\", "", text)
    text = re.sub(r"\'", "", text)
    text = re.sub(r"\"", "", text)
    return text.strip().lower()


def text_to_embedding(tokens):
    embeddings = [word_vectors[word] for word in tokens if word in word_vectors]
    if not embeddings:
        embeddings = [np.zeros(300)]
    return np.mean(embeddings, axis=0)


def load_and_preprocess_data():
    df = pd.read_csv("../datasets/tensorflow.csv")
    df = df[['Title', 'Body', 'class']]
    df['Body'] = df['Body'].fillna('')
    df['text'] = df['Title'] + " " + df['Body']
    df = df[['text', 'class']]

    df['text'] = df['text'].apply(remove_html)
    df['text'] = df['text'].apply(remove_emoji)
    df['text'] = df['text'].apply(clean_str)
    df['text'] = df['text'].apply(remove_stopwords)
    df['tokens'] = df['text'].apply(lambda x: word_tokenize(x.lower()))
    df['embeddings'] = df['tokens'].apply(text_to_embedding)

    X = np.stack(df['embeddings'].values)
    y = df['class'].values

    return X, y
