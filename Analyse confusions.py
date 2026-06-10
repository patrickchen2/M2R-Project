"""
analyse_confusions.py

Inspects the confusion matrices that DS learned for each model.

Answers:
  - Which models are most reliable?
  - Which bins are hardest to predict?
  - Are the models correlated? (tests the DS independence assumption)

Usage:
    python analyse_confusions.py
"""

import numpy as np
from Models import X_train, X_test, y_train, y_test, models
from discretisemodels import prepare_ds_input
from DawidSkeneEM import DawidSkeneEM as VanillaDS


# --- Train models and fit DS ---

for name, model in models:
    model.fit(X_train, y_train)

N_BINS = 5
annotations, model_names, item_names, class_names, bin_edges = prepare_ds_input(
    models=models, X_test=X_test, y_train=y_train, n_bins=N_BINS
)

ds = VanillaDS(max_iter=100, tol=1e-6, smoothing=1e-3)
ds.fit(annotations)

confusion_matrices = ds.worker_confusion_matrices()


# --- Print each model's confusion matrix ---
# Rows = true class, columns = predicted class, rows sum to 1

print("Confusion matrices (rows = true bin, columns = predicted bin)")
print("Each row shows: if the true temperature is in bin X, how often does")
print("the model predict each bin? Diagonal = correct prediction.")
print()

for worker, matrix in confusion_matrices.items():
    print(f"  {worker}")
    header = "          " + "".join(f"  bin{j}" for j in range(len(class_names)))
    print(header)
    for j, row in enumerate(matrix):
        row_str = "".join(f"  {v:.2f}" for v in row)
        print(f"  bin{j}   {row_str}")
    print()


# --- Reliability: diagonal score ---
# A perfect model always predicts the true bin -> diagonal is all 1s
# Average diagonal = 1/n_classes for a random model (= 0.2 for 5 bins)

print("Reliability scores (average diagonal of confusion matrix)")
print("1.0 = perfect, 0.2 = no better than random guessing")
print("-" * 45)

scores = []
for worker, matrix in confusion_matrices.items():
    score = float(np.trace(matrix) / matrix.shape[0])
    scores.append((worker, score))
    print(f"  {str(worker):<28} {score:.3f}")

scores.sort(key=lambda x: -x[1])
print("-" * 45)
print(f"  Most reliable:  {scores[0][0]}")
print(f"  Least reliable: {scores[-1][0]}")


# --- Pairwise agreement between models ---
# DS assumes models predict independently. If two models almost always agree,
# that assumption is violated and DS may be overconfident.
# We check this by looking at how often each pair predicts the same bin.

print()
print("Pairwise agreement between models (fraction of test days where")
print("both models predict the same bin)")
print("High agreement = models are correlated = DS independence assumption")
print("is more violated for that pair.")
print()

inner_edges = bin_edges[1:-1]
bin_preds = np.array([
    np.digitize(model.predict(X_test), inner_edges)
    for _, model in models
])  # shape (n_models, n_test)

header = f"{'':22}" + "".join(f"{name[:8]:>10}" for name in model_names)
print(header)
for i in range(len(model_names)):
    row = f"  {model_names[i][:20]:<20}"
    for j in range(len(model_names)):
        agreement = np.mean(bin_preds[i] == bin_preds[j])
        row += f"  {agreement:.3f}   "
    print(row)