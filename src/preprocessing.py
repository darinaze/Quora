import re
import numpy as np

def preprocess(text):
    text = text.lower()
    tokens = re.findall(r"[a-z]+", text)

    processed_tokens = []
    for token in tokens:
        processed_tokens.append(token)

    return processed_tokens


def fill_missing_questions(df):
    df["question1"] = df["question1"].fillna("")
    df["question2"] = df["question2"].fillna("")
    return df


def cosine_similarity_pairs(X_q1, X_q2):
    """
    Row-wise cosine similarity for TF-IDF matrices
    """
    cos = X_q1.multiply(X_q2).sum(axis=1)
    return np.array(cos).ravel()

def count_common_words(question1, question2):
    words_in_question1 = question1.split()
    words_in_question2 = question2.split()

    unique_words_question1 = set(words_in_question1)
    unique_words_question2 = set(words_in_question2)

    common_words = unique_words_question1.intersection(unique_words_question2)

    return len(common_words)

import pandas as pd

def common_words_count_for_df(df):
    
    common_words_counts = []

    for i in range(len(df)):
        question1 = str(df.loc[i, "question1"])
        question2 = str(df.loc[i, "question2"])

        count = count_common_words(question1, question2)
        common_words_counts.append(count)

    return pd.Series(common_words_counts)

import numpy as np

def cosine_similarity_between_embeddings(embeddings_q1, embeddings_q2):
   
    elementwise_product = embeddings_q1 * embeddings_q2

    similarity_numerator = np.sum(elementwise_product, axis=1)

    q1_vector_length = np.linalg.norm(embeddings_q1, axis=1)
    q2_vector_length = np.linalg.norm(embeddings_q2, axis=1)

    similarity_denominator = (q1_vector_length * q2_vector_length) + 1e-9

    cosine_similarities = similarity_numerator / similarity_denominator

    return cosine_similarities

def normalize_embedding_vectors(embedding_vectors, small_number=1e-9):
    """
    Робить всі embedding-вектори однакової довжини.

    Навіщо:
    - SBERT може давати вектори різної довжини
    - нам важливий сенс, а не масштаб
    - після нормалізації порівняння стає стабільнішим
    """

    vector_lengths = np.linalg.norm(embedding_vectors, axis=1, keepdims=True)

    normalized_vectors = embedding_vectors / (vector_lengths + small_number)

    return normalized_vectors
