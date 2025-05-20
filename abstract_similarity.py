# TUGAS KELOMPOK
# RAFAEL ANDIKA RADYA HADI KUSUMA
# IGNATIUS GERALD HANDONO

import math
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import cosine_similarity

# (if needed)
# import nltk
# nltk.download('punkt')
# nltk.download('stopwords')


# Term Frequency
def tf(words, word):
    return words.count(word) / len(words) if words else 0


# Inverse Document Frequency with smoothing
def idf(N, df):
    return math.log((N + 1) / (df + 1)) + 1


# Get vocabulary
def get_vocabulary(tokenized_texts):
    vocabulary = set()
    for tokens in tokenized_texts:
        vocabulary.update(tokens)
    return list(vocabulary)


# TF-IDF Vectorizer
def tf_idf_vectorize(texts, remove_stopwords=True, lowercase=True):
    if lowercase:
        texts = [text.lower() for text in texts]

    # Tokenization
    tokenized = [word_tokenize(text) for text in texts]

    # Remove stopwords
    if remove_stopwords:
        stop_words = set(stopwords.words('english'))
        tokenized = [
            [word for word in doc if word.isalpha() and word not in stop_words]
            for doc in tokenized
        ]
    else:
        tokenized = [
            [word for word in doc if word.isalpha()]
            for doc in tokenized
        ]

    # Check for empty documents
    for i, doc in enumerate(tokenized):
        if not doc:
            print(f"Warning: Document {i+1} is empty after preprocessing!")

    vocabulary = get_vocabulary(tokenized)

    # Document frequency
    df = {}
    for doc in tokenized:
        for word in set(doc):
            df[word] = df.get(word, 0) + 1

    # TF-IDF vectors
    tfidf_vectors = []
    N = len(tokenized)
    for doc in tokenized:
        vector = []
        for word in vocabulary:
            tf_score = tf(doc, word)
            idf_score = idf(N, df.get(word, 0))
            vector.append(tf_score * idf_score)
        tfidf_vectors.append(vector)

    return tfidf_vectors, vocabulary


# Cosine similarity between two vectors
def cosine_similarity_pair(x, y):
    dot = sum(a * b for a, b in zip(x, y))
    norm_x = math.sqrt(sum(a ** 2 for a in x))
    norm_y = math.sqrt(sum(b ** 2 for b in y))
    if norm_x == 0 or norm_y == 0:
        return 0.0
    return dot / (norm_x * norm_y)


# Cosine similarity matrix
def cosine_similarity_matrix(tf_idf_vectors):
    array = np.array(tf_idf_vectors)
    covariance_mat = np.matmul(array, np.transpose(array))
    D = np.diag(1 / np.sqrt(np.diag(covariance_mat)))
    return D@covariance_mat@D


# Visualization
def plot_similarity_heatmap(sim_matrix, labels):
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        sim_matrix,
        annot=True,
        cmap='Blues',
        xticklabels=labels,
        yticklabels=labels
    )
    plt.title("Cosine Similarity Heatmap")
    plt.tight_layout()
    plt.savefig("cosine_similarity_heatmap.png")
    plt.close()


# Main Program
if __name__ == "__main__":
    try:
        # Load documents
        with open("abstract1.txt", "r") as f1, \
             open("abstract2.txt", "r") as f2, \
             open("abstract3.txt", "r") as f3:
            documents = [f1.read(), f2.read(), f3.read()]
            labels = ["Doc1", "Doc2", "Doc3"]

        # Compute TF-IDF vectors
        tfidf_vectors, vocab = tf_idf_vectorize(documents)

        print("\nTF-IDF Vectors:")
        for i, vec in enumerate(tfidf_vectors):
            print(f"{labels[i]}: {vec}")

        # Pairwise cosine similarity
        print("\nPairwise Cosine Similarities:")
        sim_01 = cosine_similarity_pair(tfidf_vectors[0], tfidf_vectors[1])
        print(f"{labels[0]} vs {labels[1]}: {sim_01:.4f}")
        sim_02 = cosine_similarity_pair(tfidf_vectors[0], tfidf_vectors[2])
        print(f"{labels[0]} vs {labels[2]}: {sim_02:.4f}")
        sim_12 = cosine_similarity_pair(tfidf_vectors[1], tfidf_vectors[2])
        print(f"{labels[1]} vs {labels[2]}: {sim_12:.4f}")

        # Full similarity matrix
        sim_matrix = cosine_similarity_matrix(tfidf_vectors)
        print("\nCosine Similarity Matrix:")
        print(sim_matrix)

        # Save heatmap
        plot_similarity_heatmap(sim_matrix, labels)
        print("\nCosine similarity heatmap saved.")

    except FileNotFoundError as e:
        print(f"Error: {e}.")
