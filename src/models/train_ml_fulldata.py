# File: src/models/train_ml_fulldata.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


def get_dataset():
    s1 = pd.read_csv('outputs/session1.csv')
    s2 = pd.read_csv('outputs/session2.csv')
    s3 = pd.read_csv('outputs/session3.csv')
    df = pd.concat([s1, s2, s3], ignore_index=True)
    y = df['label'].values
    X = df.drop('label', axis=1).values
    return X, y


def get_models():
    return {
        'Decision Tree': DecisionTreeClassifier(),
        'K-Nearest Neighbor': KNeighborsClassifier(),
        'Gaussian Naive Bayes': GaussianNB(),
        'Random Forest': RandomForestClassifier()
    }


def train_and_evaluate():
    X, y = get_dataset()
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)

    models = get_models()
    for name, model in models.items():
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)

        print(f"\nModel: {name}")
        print(classification_report(y_test, y_pred))
        print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%\n")


if __name__ == "__main__":
    train_and_evaluate()
