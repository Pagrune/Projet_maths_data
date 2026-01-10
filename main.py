import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_digits
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler


from models_scratch import train_scratch, predict as predict_scratch, lasso_reg
from models_scikit import train_sklearn

# Preprocessing 

digits = load_digits()
X, y = digits.data, digits.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Entrainement des modèles
print("Entraînement du modèle From Scratch...")
W, b, losses = train_scratch(X_train, y_train, X_train.shape[1], 10)

# Affichage de la courbe de l'entropy croisée loss en fonction des époques
plt.plot(losses)
plt.xlabel("Epoch")
plt.ylabel("Cross-entropy loss")
plt.show()

print("Entraînement du modèle Scikit-Learn...")
clf = train_sklearn(X_train, y_train)


y_pred_scratch = np.argmax(predict_scratch(X_test, W, b), axis=1)
y_pred_sklearn = clf.predict(X_test)

acc_scratch = accuracy_score(y_test, y_pred_scratch)
acc_sklearn = accuracy_score(y_test, y_pred_sklearn)

print(f"\nPrécision Scratch : {acc_scratch:.4f}")
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


print("\nCalcul du chemin Lasso sur les données Digits (Pixels)...")

alphas, coefs = lasso_reg(X_train, y_train)

# Affichage du graphique
plt.figure(figsize=(10, 6))
for i in range(coefs.shape[1]):
    # On ne met une légende que pour quelques pixels clés pour éviter de saturer le graph
    label = f"Pixel {i}" if i % 10 == 0 else "" 
    plt.plot(alphas, coefs[:, i], label=label, alpha=0.7)

plt.xscale('log')
plt.xlabel('Alpha (Force de la régularisation)')
plt.ylabel('Valeur des Coefficients (Poids des pixels)')
plt.title('Importance des pixels : Chemin de régularisation Lasso')
plt.axhline(0, color='black', linestyle='--', alpha=0.3)
plt.legend(loc='upper right', fontsize='small', ncol=2)
plt.grid(True, which="both", ls="-", alpha=0.2)
plt.show()

# BONUS : Visualiser quels pixels le Lasso garde à un alpha moyen
lasso_intermediaire = Lasso(alpha=0.01).fit(X_train, y_train)
pixels_importants = lasso_intermediaire.coef_.reshape(8, 8)



plt.figure(figsize=(4, 4))
plt.imshow(pixels_importants, cmap='RdBu', interpolation='nearest')
plt.title("Carte de chaleur des pixels\nretenus par Lasso (alpha=0.01)")
plt.colorbar()
plt.show()