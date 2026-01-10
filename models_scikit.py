from sklearn.linear_model import LogisticRegression

def train_sklearn(X_train, y_train):
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)
    return clf