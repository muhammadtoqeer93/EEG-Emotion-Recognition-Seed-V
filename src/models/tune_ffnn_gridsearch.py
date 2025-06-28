# File: src/models/tune_ffnn_gridsearch.py

import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier


def get_dataset(path='outputs/session1.csv'):
    df = pd.read_csv(path)
    y = df['label'].values
    X = df.drop('label', axis=1).values
    X = MinMaxScaler().fit_transform(X)
    return X, y


def get_model(units=16, learning_rate=0.01):
    model = Sequential([
        Input(shape=(5,), name="Input-Layer"),
        Dense(units, activation="relu", name="Hidden-Layer-1"),
        Dense(units//2, activation="relu", name="Hidden-Layer-2"),
        Dense(1, activation="sigmoid", name="Output-Layer")
    ])
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model


def perform_grid_search():
    X, y = get_dataset()
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)

    model = KerasClassifier(build_fn=get_model, verbose=0, epochs=10)
    param_grid = {
        'batch_size': [16, 32],
        'units': [10, 20],
        'learning_rate': [0.01, 0.05]
    }

    grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=1)
    result = grid.fit(x_train, y_train)

    print(f"Best Score: {result.best_score_:.4f}")
    print(f"Best Params: {result.best_params_}")


if __name__ == "__main__":
    perform_grid_search()
