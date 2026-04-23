import marimo

__generated_with = "0.23.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score

    return LogisticRegression, accuracy_score, np, train_test_split


@app.cell
def _(np):
    data = np.load('colon_data.npz')
    return (data,)


@app.cell
def _(data):
    data.keys()
    return


@app.cell
def _(data):
    data['bag_id']
    return


@app.cell
def _(data, train_test_split):
    X_train, X_test, y_train, y_test = train_test_split(
        data['X'], data['y'],
        test_size=0.2,
        random_state=42
    )
    return X_test, X_train, y_test, y_train


@app.cell
def _(LogisticRegression, X_train, y_train):
    clf = LogisticRegression(
        penalty='l1',
        solver='liblinear',
        C=1.0,
        max_iter=1000,
        random_state=42
    )
    clf.fit(X_train, y_train)
    return (clf,)


@app.cell
def _(X_test, accuracy_score, clf, y_test):
    y_pred = clf.predict(X_test)
    test_error = 1 - accuracy_score(y_test, y_pred)
    print(f'Test error: {test_error:.4f}')
    return


@app.cell
def _(clf, np):
    selected_features = np.sum(clf.coef_ != 0)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
