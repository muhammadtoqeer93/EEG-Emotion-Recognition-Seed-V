# File: src/evaluation/evaluate_ml_crossval.py

import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier


def get_dataset(path='outputs/session1.csv'):
    df = pd.read_csv(path)
    y = df['label'].values
    X = df.drop('label', axis=1).values
    return X, y


def get_models():
    return {
        'Decision Tree': DecisionTreeClassifier(),
        'K-Nearest Neighbor': KNeighborsClassifier(),
        'Gaussian Naive Bayes': GaussianNB()
    }


def evaluate_models(cv=10):
    X, y = get_dataset()
    models = get_models()
    
    for name, model in models.items():
        scores = cross_val_score(model, X, y, cv=cv)
        print(f"\nModel: {name}")
        print(f"Scores: {scores}")
        print(f"Mean Accuracy: {scores.mean():.2f}, Std Dev: {scores.std():.2f}")


if __name__ == "__main__":
    evaluate_models()
