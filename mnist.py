from sklearn.datasets import fetch_openml
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from models_scratch import train_scratch, predict as predict_scratch, find_best_lambda_l2
from models_scikit import train_sklearn

# -------------------------------
# 1️⃣ Chargement MNIST
# -------------------------------
mnist = fetch_openml("mnist_784", version=1, as_frame=False)
X = mnist.data      
y = mnist.target.astype(int)

# Train / test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=10000, random_state=42, stratify=y
)

# Normalisation

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Réduction pour Scratch 
X_train_small = X_train_scaled[:30000]
y_train_small = y_train[:30000] 

n_features = X_train.shape[1]  
n_classes = 10

# Scratch sans régularisation

W, b, losses = train_scratch(X_train_small, y_train_small, n_features, n_classes,
                             lr=0.01, epochs=500, lambda_reg=0)

# Scratch avec régularisation

lambda_reg = 0.001 
W_L2, b_L2, losses_L2 = train_scratch(X_train_small, y_train_small, n_features, n_classes,
                                      lr=0.01, epochs=500, lambda_reg=lambda_reg)


clf_log = train_sklearn(X_train_small, y_train_small)


clf_svc = SVC(kernel='linear', C=1.0)
clf_svc.fit(X_train_small, y_train_small)


clf_rf = RandomForestClassifier(n_estimators=100, random_state=42)
clf_rf.fit(X_train_small, y_train_small)


y_pred_scratch = np.argmax(predict_scratch(X_test_scaled, W, b), axis=1)
y_pred_L2 = np.argmax(predict_scratch(X_test_scaled, W_L2, b_L2), axis=1)
y_pred_log = clf_log.predict(X_test_scaled)
y_pred_svc = clf_svc.predict(X_test_scaled)
y_pred_rf  = clf_rf.predict(X_test_scaled)


#Accuracy
print("Scratch:", accuracy_score(y_test, y_pred_scratch))
print("Scratch + L2:", accuracy_score(y_test, y_pred_L2))
print("LogisticRegression:", accuracy_score(y_test, y_pred_log))
print("SVC:", accuracy_score(y_test, y_pred_svc))
print("RandomForest:", accuracy_score(y_test, y_pred_rf))


plt.figure(figsize=(7, 5))
plt.plot(losses, label="Scratch")
plt.plot(losses_L2, label="Scratch + L2", linestyle="--")
plt.xlabel("Epochs")
plt.ylabel("Cross-entropy loss")
plt.title("Évolution de la loss")
plt.legend()
plt.grid(True)
plt.show()


acc_scratch = accuracy_score(y_test, y_pred_scratch)
acc_sklearn_rf = accuracy_score(y_test, y_pred_rf)

cm_scratch = confusion_matrix(y_test, y_pred_scratch)
cm_sklearn_rf = confusion_matrix(y_test, y_pred_rf)

# Affichage des matrices de confusion
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

sns.heatmap(cm_scratch, annot=True, fmt="d", cmap="Blues", ax=ax1)
ax1.set_title(f"Matrice de confusion - Scratch\n(Acc: {acc_scratch:.4f})")
ax1.set_ylabel("Vraie classe")
ax1.set_xlabel("Classe prédite")

sns.heatmap(cm_sklearn_rf, annot=True, fmt="d", cmap="Greens", ax=ax2)
ax2.set_title(f"Matrice de confusion - Scikit-Learn\n(Acc: {acc_sklearn_rf:.4f})")
ax2.set_ylabel("Vraie classe")
ax2.set_xlabel("Classe prédite")

plt.tight_layout()
plt.show()
