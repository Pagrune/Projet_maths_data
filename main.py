import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_digits
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler


from models_scratch import train_scratch, predict as predict_scratch, find_best_lambda_l2
from models_scikit import train_sklearn

# Preprocessing 

digits = load_digits()
X, y = digits.data, digits.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Entrainement des modèles
print("Entraînement du modèle From Scratch sans régularisation ...")
W, b, losses = train_scratch(X_train, y_train, X_train.shape[1], 10)


# Essai pour trouver le meilleur lambda

print("\n Test pour trouver le meilleur lambda pour la régularisation L2")

lambdas = [0, 0.0001, 0.001, 0.01, 0.1, 1, 10]

best_lambda, scores = find_best_lambda_l2(
    X_train, y_train,
    X_test, y_test,   # utilisé comme validation
    n_classes=10,
    lambdas=lambdas
)

print("Best lambda =", best_lambda)


print("Entraînement du modèle From Scratch avec régularisation L2...")
W_L2, b_L2, losses_L2 = train_scratch(
    X_train, y_train,
    X_train.shape[1], 10,
    lr=0.01,
    epochs=1000,
    lambda_reg=best_lambda
)


plt.figure(figsize=(7, 5))

plt.plot(losses, label="Sans régularisation (Scratch)")
plt.plot(losses_L2, label="Avec L2 (Scratch)", linestyle="--")

plt.xlabel("Epochs")
plt.ylabel("Cross-entropy loss")
plt.title("Évolution de la loss")
plt.legend()
plt.grid(True)
plt.show()


print("Entraînement du modèle Scikit-Learn...")
clf = train_sklearn(X_train, y_train)


y_pred_scratch = np.argmax(predict_scratch(X_test, W, b), axis=1)
y_pred_L2 = np.argmax(predict_scratch(X_test, W_L2, b_L2), axis=1)
y_pred_sklearn = clf.predict(X_test)

acc_scratch = accuracy_score(y_test, y_pred_scratch)
acc_L2 = accuracy_score(y_test, y_pred_L2)
acc_sklearn = accuracy_score(y_test, y_pred_sklearn)

print(f"\nPrécision Scratch : {acc_scratch:.4f}")
print(f"Accuracy Scratch AVEC L2 : {acc_L2:.4f}")
print(f"Précision Sklearn : {acc_sklearn:.4f}")



# Matrices de confusion

cm_scratch = confusion_matrix(y_test, y_pred_scratch)
cm_sklearn = confusion_matrix(y_test, y_pred_sklearn)


# Affichage des matrices de confusion
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

sns.heatmap(cm_scratch, annot=True, fmt="d", cmap="Blues", ax=ax1)
ax1.set_title(f"Matrice de confusion - Scratch\n(Acc: {acc_scratch:.4f})")
ax1.set_ylabel("Vraie classe")
ax1.set_xlabel("Classe prédite")

sns.heatmap(cm_sklearn, annot=True, fmt="d", cmap="Greens", ax=ax2)
ax2.set_title(f"Matrice de confusion - Scikit-Learn\n(Acc: {acc_sklearn:.4f})")
ax2.set_ylabel("Vraie classe")
ax2.set_xlabel("Classe prédite")

plt.tight_layout()
plt.show()



# Visualisation des différents poids pour mon modèle from scratch et celui de scikit-learn
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
