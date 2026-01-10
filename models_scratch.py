import numpy as np
from sklearn.linear_model import Lasso


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

def train_scratch(X_train, y_train, n_features, n_classes, lr=0.01, epochs=1000, lambda_reg=0.001):
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


def lasso_reg(X, y):
    alphas = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000]
    coefs_lasso = []
    
    for alpha in alphas:
        lasso = Lasso(alpha=alpha, max_iter=10000)
        lasso.fit(X, y)
        # Attention : .coef_ avec un underscore
        coefs_lasso.append(lasso.coef_.flatten())
        
    return alphas, np.array(coefs_lasso)