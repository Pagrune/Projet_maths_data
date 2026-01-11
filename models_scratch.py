import numpy as np
from sklearn.linear_model import Lasso, Ridge
from sklearn.metrics import accuracy_score


def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

def predict(X, W, b):
    logits = X @ W + b
    return softmax(logits)

def crossEntropy(y_true, y_pred):
    n = y_true.shape[0]
    return -np.sum(np.log(y_pred[np.arange(n), y_true] + 1e-15)) / n

def gradients(X, y, probs):
    n = X.shape[0]
    Y = np.zeros_like(probs)
    Y[np.arange(n), y] = 1 
    dW = X.T @ (probs - Y) / n  
    db = np.mean(probs - Y, axis=0)
    return dW, db

def train_scratch(X_train, y_train, n_features, n_classes, lr=0.01, epochs=1000, lambda_reg=0):
    W = np.random.randn(n_features, n_classes) * 0.01
    b = np.zeros(n_classes)
    losses = []
    
    for epoch in range(epochs):
        probs = predict(X_train, W, b)
        loss = crossEntropy(y_train, probs) + (lambda_reg / 2) * np.sum(W**2)
        losses.append(loss)
        
        dW, db = gradients(X_train, y_train, probs)
        dW_reg = dW + lambda_reg * W
        
        W -= lr * dW_reg
        b -= lr * db

        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss = {loss:.4f}")

    return W, b, losses

def find_best_lambda_l2(X_train, y_train, X_val, y_val, n_classes, lambdas):
    scores = []

    for l in lambdas:
        W, b, _ = train_scratch(
            X_train, y_train,
            n_features=X_train.shape[1],
            n_classes=n_classes,
            lr=0.01,
            epochs=500,
            lambda_reg=l
        )

        # prédiction sur validation
        probs = predict(X_val, W, b)
        y_pred = np.argmax(probs, axis=1)

        acc = accuracy_score(y_val, y_pred)
        scores.append(acc)

    best_idx = np.argmax(scores)
    best_lambda = lambdas[best_idx]

    return best_lambda, scores

def train_scratch_stoc(X_train, y_train, n_features, n_classes, lr=0.01, epochs=1000, lambda_reg=0):
    W = np.random.randn(n_features, n_classes) * 0.01
    b = np.zeros(n_classes)
    losses = []
    n = X_train.shape[0]

    for epoch in range(epochs):
        indices = np.random.permutation(n)

        for i in indices:
            x_i = X_train[i:i+1]  
            y_i = y_train[i]   

            # ► forward
            probs = predict(x_i, W, b) 

            loss_i = -np.log(probs[0, y_i] + 1e-15)

            Y = np.zeros_like(probs)
            Y[0, y_i] = 1

            dW = x_i.T @ (probs - Y) + lambda_reg * W
            db = (probs - Y).ravel()

            W -= lr * dW
            b -= lr * db

        probs_all = predict(X_train, W, b)
        loss_epoch = crossEntropy(y_train, probs_all) + (lambda_reg / 2) * np.sum(W**2)
        losses.append(loss_epoch)

        if epoch % 1 == 0:
            print(f"Epoch {epoch}, Loss = {loss_epoch:.4f}")

    return W, b, losses
