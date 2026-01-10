import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

# Import de vos modules
from models_scratch import train_scratch, predict as predict_scratch
from models_scikit import train_sklearn

# 1. DATA PREPARATION
digits = load_digits()
X, y = digits.data, digits.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 2. ENTRAÎNEMENT
print("Entraînement du modèle From Scratch...")
W, b, losses = train_scratch(X_train, y_train, X_train.shape[1], 10)

print("Entraînement du modèle Scikit-Learn...")
clf = train_sklearn(X_train, y_train)

# 3. ÉVALUATION ET COMPARAISON
y_pred_scratch = np.argmax(predict_scratch(X_test, W, b), axis=1)
y_pred_sklearn = clf.predict(X_test)

acc_scratch = accuracy_score(y_test, y_pred_scratch)
acc_sklearn = accuracy_score(y_test, y_pred_sklearn)

print(f"\nPrécision Scratch : {acc_scratch:.4f}")
print(f"Précision Sklearn : {acc_sklearn:.4f}")

# 4. VISUALISATION DES POIDS
def plot_weights_comparison(W_scratch, W_sklearn):
    fig, axes = plt.subplots(4, 5, figsize=(15, 10))
    for i in range(10):
        # Scratch
        ax_s = axes[i // 5, i % 5]
        ax_s.imshow(W_scratch[:, i].reshape(8, 8), cmap='RdBu')
        ax_s.set_title(f"Scratch {i}")
        ax_s.axis('off')
        # Sklearn
        ax_sk = axes[(i // 5) + 2, i % 5]
        ax_sk.imshow(W_sklearn[i, :].reshape(8, 8), cmap='RdBu')
        ax_sk.set_title(f"Sklearn {i}")
        ax_sk.axis('off')
    plt.tight_layout()
    plt.show()

plot_weights_comparison(W, clf.coef_)