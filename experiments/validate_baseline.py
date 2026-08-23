
from src.gblistsvm import GBLSTSVM

# Baseline Validation Script for GBLSTSVM
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, balanced_accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Generate synthetic data
X = np.random.randn(1000, 20)
Y = np.random.choice([0, 1], size=1000)

# Split data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

# Scale data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train GBLSTSVM model
model = GBLSTSVM(c1=1.0, c2=0.1, max_iter=1000)
model.fit(X_train, Y_train)

# Predict
Y_pred = model.predict(X_test)

# Evaluate
print(f"Accuracy: {accuracy_score(Y_test, Y_pred):.4f}")
print(f"F1 Score: {f1_score(Y_test, Y_pred):.4f}")
print(f"AUROC: {roc_auc_score(Y_test, Y_pred):.4f}")
print(f"Balanced Accuracy: {balanced_accuracy_score(Y_test, Y_pred):.4f}")