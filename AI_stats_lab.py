"""
AIstats_lab.py

Student starter file for:
1. Naive Bayes spam classification
2. K-Nearest Neighbors on Iris
"""

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


def accuracy_score(y_true, y_pred):
    """
    Compute classification accuracy.
    """
    return float(np.mean(y_true == y_pred))


# =========================
# Q1 Naive Bayes
# =========================

def naive_bayes_mle_spam():

    texts = [
        "win money now",
        "limited offer win cash",
        "cheap meds available",
        "win big prize now",
        "exclusive offer buy now",
        "cheap pills buy cheap meds",
        "win lottery claim prize",
        "urgent offer win money",
        "free cash bonus now",
        "buy meds online cheap",
        "meeting schedule tomorrow",
        "project discussion meeting",
        "please review the report",
        "team meeting agenda today",
        "project deadline discussion",
        "review the project document",
        "schedule a meeting tomorrow",
        "please send the report",
        "discussion on project update",
        "team sync meeting notes"
    ]

    labels = np.array([
        1,1,1,1,1,1,1,1,1,1,
        0,0,0,0,0,0,0,0,0,0
    ])

    test_email = "win cash prize now"

    # Tokenize
    tokenized = [text.split() for text in texts]

    # Vocabulary
    vocab = set(word for sent in tokenized for word in sent)

    # Priors
    priors = {
        0: np.mean(labels == 0),
        1: np.mean(labels == 1)
    }

    # Word counts per class
    word_counts = {0: {}, 1: {}}
    total_words = {0: 0, 1: 0}

    for sent, label in zip(tokenized, labels):
        for word in sent:
            word_counts[label][word] = word_counts[label].get(word, 0) + 1
            total_words[label] += 1

    # Word probabilities
    word_probs = {0: {}, 1: {}}

    for c in [0, 1]:
        for word in vocab:
            count = word_counts[c].get(word, 0)
            word_probs[c][word] = count / total_words[c]  # MLE

    # Prediction
    test_words = test_email.split()

    scores = {}

    for c in [0, 1]:
        score = priors[c]
        for word in test_words:
            score *= word_probs[c].get(word, 0)
        scores[c] = score

    prediction = max(scores, key=scores.get)

    return priors, word_probs, prediction

# =========================
# Q2 KNN
# =========================

def knn_iris(k=3, test_size=0.2, seed=0):

    # Load dataset
    data = load_iris()
    X = data.data
    y = data.target

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed
    )

    # Distance function
    def euclidean(a, b):
        return np.sqrt(np.sum((a - b) ** 2))

    # Predict function
    def predict(X1, X_train, y_train, k):
        preds = []

        for x in X1:
            distances = []

            for i in range(len(X_train)):
                d = euclidean(x, X_train[i])
                distances.append((d, y_train[i]))

            distances.sort(key=lambda x: x[0])

            neighbors = [label for _, label in distances[:k]]

            # Majority vote
            counts = np.bincount(neighbors)
            pred = np.argmax(counts)

            preds.append(pred)

        return np.array(preds)

    # Predictions
    train_preds = predict(X_train, X_train, y_train, k)
    test_preds = predict(X_test, X_train, y_train, k)

    # Accuracy
    train_accuracy = accuracy_score(y_train, train_preds)
    test_accuracy = accuracy_score(y_test, test_preds)

    return train_accuracy, test_accuracy, test_preds
