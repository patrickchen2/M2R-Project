"""
evaluate.py

Compares all DS methods on the London weather test set.
"""

import numpy as np
from sklearn.metrics import mean_absolute_error

from Models import X_train, X_test, y_train, y_test, models
from discretisemodels import make_temperature_bins, prepare_ds_input, prepare_soft_ds_input
from DawidSkeneEM import DawidSkeneEM as VanillaDS
from DSspectral import DawidSkeneEM as SpectralDS
from SoftDawidSkene import SoftDawidSkene

for name, model in models:
    model.fit(X_train, y_train)


# --- Helper: convert DS bin prediction back to a temperature ---

def bin_midpoints(bin_edges):
    inner = bin_edges[1:-1]
    width = (inner[-1] - inner[0]) / max(len(inner) - 1, 1)
    midpoints = []
    for i in range(len(bin_edges) - 1):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        if np.isinf(lo): lo = hi - width
        if np.isinf(hi): hi = lo + width
        midpoints.append(0.5 * (lo + hi))
    return np.array(midpoints)

def preds_to_temps(predict_dict, item_names, class_names, midpoints):
    preds = []
    for item in item_names:
        idx = class_names.index(predict_dict[item])
        preds.append(midpoints[idx])
    return np.array(preds)


# --- Evaluate everything ---

N_BINS = 5
results = {}

# Individual models
for name, model in models:
    mae = mean_absolute_error(y_test, model.predict(X_test))
    results[name] = mae

# Ensemble average
ensemble_preds = np.mean([m.predict(X_test) for _, m in models], axis=0)
results["Ensemble average"] = mean_absolute_error(y_test, ensemble_preds)

# Shared DS setup
annotations, model_names, item_names, class_names, bin_edges = prepare_ds_input(
    models=models, X_test=X_test, y_train=y_train, n_bins=N_BINS
)
midpoints = bin_midpoints(bin_edges)

# Vanilla DS
ds = VanillaDS(max_iter=100, tol=1e-6, smoothing=1e-3)
ds.fit(annotations)
results["Vanilla DS"] = mean_absolute_error(
    y_test, preds_to_temps(ds.predict(), item_names, class_names, midpoints)
)

# Spectral DS
ds_s = SpectralDS(init="spectral", max_iter=100, tol=1e-6, smoothing=1e-3, random_state=42)
ds_s.fit(annotations)
results["Spectral DS"] = mean_absolute_error(
    y_test, preds_to_temps(ds_s.predict(), item_names, class_names, midpoints)
)

# Soft DS
probs, mn, in_, cn, _, _ = prepare_soft_ds_input(
    models=models, X_train=X_train, y_train=y_train, X_test=X_test, n_bins=N_BINS
)
sds = SoftDawidSkene(max_iter=200, polyak_alpha=1e-2, m_steps=5, lr=1e-4, verbose=False)
sds.fit(probs, item_names=in_, model_names=mn, class_names=cn)
results["Soft DS"] = mean_absolute_error(
    y_test, preds_to_temps(sds.predict(), in_, cn, midpoints)
)


# --- Print results ---

print("\nMAE comparison (degrees C, lower is better)")
print("-" * 40)
for name, mae in sorted(results.items(), key=lambda x: x[1]):
    print(f"  {name:<28} {mae:.3f}")