"""
evaluate.py

Compares all DS methods on the London weather test set.
"""

import numpy as np
from sklearn.metrics import mean_absolute_error

from Models import X_train, X_test, y_train, y_test, models
from discretisemodels import make_temperature_bins, prepare_ds_input, prepare_soft_ds_input
from DawidSkeneEM import DawidSkeneEM as VanillaDS
from DSspectral import DawidSkeneEM as SpectralDS, ManualPartitionDS
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

# --- Helper: For KL divergence evaluation ---
def true_bin_indices(y, bin_edges):
    """
    Convert true continuous temperatures into bin indices.
    If there are 5 bins, this returns values in {0,1,2,3,4}.
    """
    return np.searchsorted(bin_edges[1:-1], y, side="right")


def proba_dict_to_array(proba_dict, item_names, class_names, eps=1e-12):
    """
    Convert predict_proba() dictionary output into an array of shape:
        (n_items, n_classes)
    """
    arr = np.array([
        [proba_dict[item].get(cls, 0.0) for cls in class_names]
        for item in item_names
    ], dtype=float)

    arr = np.clip(arr, eps, None)
    arr /= arr.sum(axis=1, keepdims=True)

    return arr


def nll_from_probs(probs, true_bins, eps=1e-12):
    """
    Negative log likelihood of the correct bin.
    Equivalent to KL divergence from a one-hot true distribution.
    """
    probs = np.asarray(probs, dtype=float)
    probs = np.clip(probs, eps, 1.0)
    probs /= probs.sum(axis=1, keepdims=True)

    return -np.mean(np.log(probs[np.arange(len(true_bins)), true_bins]))

# --- Prints the evaluation summary for the error distributions ---


def error_summary(y_true, y_pred):
    errors = y_pred - y_true
    abs_errors = np.abs(errors)

    return {
        "mean_error_bias": np.mean(errors),
        "std_error": np.std(errors),
        "median_abs_error": np.median(abs_errors),
        "p90_abs_error": np.percentile(abs_errors, 90),
        "max_abs_error": np.max(abs_errors),
    }

# --- Evaluate everything ---

N_BINS = 5
results = {}
predictions = {}

# Individual models
for name, model in models:
    pred = model.predict(X_test)
    predictions[name] = pred
    results[name] = mean_absolute_error(y_test, pred)

# Ensemble average
ensemble_preds = np.mean([m.predict(X_test) for _, m in models], axis=0)
predictions["Ensemble average"] = ensemble_preds
results["Ensemble average"] = mean_absolute_error(y_test, ensemble_preds)


# Shared DS setup
annotations, model_names, item_names, class_names, bin_edges = prepare_ds_input(
    models=models, X_test=X_test, y_train=y_train, n_bins=N_BINS
)
midpoints = bin_midpoints(bin_edges)

# get the true bin indices for KL divergence evaluation
true_bins = true_bin_indices(y_test, bin_edges)
nll_results = {}

# Vanilla DS
ds = VanillaDS(max_iter=100, tol=1e-6, smoothing=1e-3)
ds.fit(annotations)
results["Vanilla DS"] = mean_absolute_error(
    y_test, preds_to_temps(ds.predict(), item_names, class_names, midpoints)
)

vanilla_probs = proba_dict_to_array(
    ds.predict_proba(),
    item_names,
    class_names
)

nll_results["Vanilla DS"] = nll_from_probs(vanilla_probs, true_bins)

# Spectral DS
# Spectral DS
group_assignments = {
    "Linear regression":              0,
    "Decision tree":                  1,
    "Persistence (today = tomorrow)": 2,
    "Naive linear (correlations)":    0,
    "1-Nearest Neighbour":            1,
    "Seasonal (monthly average)":     2,
    "Sunshine threshold":             0,
    "Constant (training mean)":       1,
}

ds_s = ManualPartitionDS(
    group_assignments=group_assignments,
    init="spectral",
    max_iter=100,
    tol=1e-6,
    smoothing=1e-3,
    random_state=42,
    verbose=True,
)
ds_s.fit(annotations)
results["Spectral DS"] = mean_absolute_error(
    y_test, preds_to_temps(ds_s.predict(), item_names, class_names, midpoints)
)

spectral_probs = proba_dict_to_array(
    ds_s.predict_proba(),
    item_names,
    class_names
)

nll_results["Spectral DS"] = nll_from_probs(spectral_probs, true_bins)

# Soft DS
probs, mn, in_, cn, soft_bin_edges, _ = prepare_soft_ds_input(
    models=models, X_train=X_train, y_train=y_train, X_test=X_test, n_bins=N_BINS
)
sds = SoftDawidSkene(max_iter=200, polyak_alpha=1e-2, m_steps=5, lr=1e-4, verbose=False)
sds.fit(probs, item_names=in_, model_names=mn, class_names=cn)
results["Soft DS"] = mean_absolute_error(
    y_test, preds_to_temps(sds.predict(), in_, cn, midpoints)
)

true_bins_soft = true_bin_indices(y_test, soft_bin_edges)

soft_probs = proba_dict_to_array(
    sds.predict_proba(),
    in_,
    cn
)

nll_results["Soft DS"] = nll_from_probs(soft_probs, true_bins_soft)


# --- Evaluate error distributions ---
vanilla_temps = preds_to_temps(ds.predict(), item_names, class_names, midpoints)
predictions["Vanilla DS"] = vanilla_temps
results["Vanilla DS"] = mean_absolute_error(y_test, vanilla_temps)

spectral_temps = preds_to_temps(ds_s.predict(), item_names, class_names, midpoints)
predictions["Spectral DS"] = spectral_temps
results["Spectral DS"] = mean_absolute_error(y_test, spectral_temps)

soft_temps = preds_to_temps(sds.predict(), in_, cn, midpoints)
predictions["Soft DS"] = soft_temps
results["Soft DS"] = mean_absolute_error(y_test, soft_temps)

# --- Print results ---

print("\nMAE comparison (degrees C, lower is better)")
print("-" * 40)
for name, mae in sorted(results.items(), key=lambda x: x[1]):
    print(f"  {name:<28} {mae:.3f}")


print("\nNLL / KL comparison over temperature bins, lower is better")
print("-" * 55)
for name, nll in sorted(nll_results.items(), key=lambda x: x[1]):
    print(f"  {name:<28} {nll:.3f}")

print("\nError distribution summary")
print("-" * 80)
for name, pred in sorted(predictions.items(), key=lambda x: results[x[0]]):
    stats = error_summary(y_test, pred)
    print(
        f"{name:<28} "
        f"bias={stats['mean_error_bias']:+.3f}, "
        f"std={stats['std_error']:.3f}, "
        f"median|e|={stats['median_abs_error']:.3f}, "
        f"p90|e|={stats['p90_abs_error']:.3f}, "
        f"max|e|={stats['max_abs_error']:.3f}"
    )