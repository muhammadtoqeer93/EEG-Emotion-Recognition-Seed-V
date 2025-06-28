# File: src/models/train_ml_baseline.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
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


def train_and_evaluate():
    X, y = get_dataset()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)

    models = get_models()
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        print(f"\nModel: {name}")
        print(classification_report(y_test, y_pred))
        print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%\n")


if __name__ == "__main__":
    train_and_evaluate()
