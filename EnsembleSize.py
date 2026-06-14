# ensemble_size_experiment.py

import numpy as np
from sklearn.metrics import accuracy_score
from collections import Counter

from Models import X_train, X_test, y_train, y_test, models
from discretisemodels import prepare_ds_input, prepare_soft_ds_input
from DawidSkeneEM import DawidSkeneEM
from SoftDawidSkene import SoftDawidSkene

# ----------------------------------------------------------------
# Train all models first
# ----------------------------------------------------------------

for name, model in models:
    model.fit(X_train, y_train)

# ----------------------------------------------------------------
# Order models worst to best based on known MAE performance.
# This ordering comes from running Models.py and reading the output.
# Worst first so we can see D-S learning to discount bad models.
# ----------------------------------------------------------------

ordered_models = [
    models[0],  # Constant
    models[5],  # 1-Nearest Neighbour
    models[3],  # Precip threshold
    models[4],  # Sunshine threshold
    models[2],  # Seasonal
    models[6],  # Naive linear
    models[1],  # Persistence
    models[7],  # Decision tree
    models[8],  # Linear regression
]

N_BINS = 5


def get_true_bin_labels(y_test, bin_edges, class_names):
    """Convert actual temperatures to bin labels for evaluation."""
    indices = np.digitize(y_test, bin_edges[1:-1])
    return [class_names[i] for i in indices]


def majority_vote(annotations, item_names):
    """Simple majority vote baseline."""
    votes = {item: [] for item in item_names}
    for item, worker, label in annotations:
        votes[item].append(label)
    return {
        item: Counter(votes[item]).most_common(1)[0][0]
        for item in item_names
    }


def run_experiment_for_subset(subset, n_bins=N_BINS):
    """
    Run majority vote, DS, and SDS on a given subset of models.
    Returns accuracies for each method.
    """

    # --- Hard label setup for majority vote and DS ---
    annotations, model_names, item_names, class_names, bin_edges = (
        prepare_ds_input(
            models=subset,
            X_test=X_test,
            y_train=y_train,
            n_bins=n_bins,
        )
    )

    true_labels = get_true_bin_labels(y_test, bin_edges, class_names)

    # Majority vote
    mv_preds = majority_vote(annotations, item_names)
    mv_acc = accuracy_score(
        true_labels,
        [mv_preds[item] for item in item_names]
    )

    # Basic Dawid-Skene
    ds = DawidSkeneEM(max_iter=100, tol=1e-6, smoothing=1e-3)
    ds.fit(annotations)
    ds_preds = ds.predict()
    ds_acc = accuracy_score(
        true_labels,
        [ds_preds[item] for item in item_names]
    )

    # --- Soft label setup for SDS ---
    probs, model_names_s, item_names_s, class_names_s, bin_edges_s, sigmas = (
        prepare_soft_ds_input(
            models=subset,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            n_bins=n_bins,
        )
    )

    sds = SoftDawidSkene(
        max_iter=200,
        polyak_alpha=1e-2,
        m_steps=5,
        lr=1e-4,
        verbose=False,
    )
    sds.fit(
        probs,
        item_names=item_names_s,
        model_names=model_names_s,
        class_names=class_names_s,
    )

    sds_preds = sds.predict()
    sds_acc = accuracy_score(
        true_labels,
        [sds_preds[item] for item in item_names_s]
    )

    return mv_acc, ds_acc, sds_acc


# ----------------------------------------------------------------
# Run experiment for increasing ensemble sizes
# ----------------------------------------------------------------

# Start from 2 models, go up to all 9
ensemble_sizes = list(range(2, len(ordered_models) + 1))

results = []

for size in ensemble_sizes:
    subset = ordered_models[:size]
    subset_names = [name for name, _ in subset]

    print(f"\nEnsemble size {size}: {subset_names}")

    mv_acc, ds_acc, sds_acc = run_experiment_for_subset(subset)

    results.append({
        "size": size,
        "models": subset_names,
        "majority_vote": mv_acc,
        "dawid_skene": ds_acc,
        "soft_dawid_skene": sds_acc,
    })

    print(f"  Majority Vote : {mv_acc:.3f}")
    print(f"  Dawid-Skene  : {ds_acc:.3f}")
    print(f"  Soft DS      : {sds_acc:.3f}")

# ----------------------------------------------------------------
# Print summary table
# ----------------------------------------------------------------

print("\n" + "="*60)
print(f"{'Size':<6} {'MV':>8} {'DS':>8} {'SDS':>8} {'DS gain':>10} {'SDS gain':>10}")
print("="*60)

for r in results:
    ds_gain = r["dawid_skene"] - r["majority_vote"]
    sds_gain = r["soft_dawid_skene"] - r["majority_vote"]
    print(
        f"{r['size']:<6} "
        f"{r['majority_vote']:>8.3f} "
        f"{r['dawid_skene']:>8.3f} "
        f"{r['soft_dawid_skene']:>8.3f} "
        f"{ds_gain:>+10.3f} "
        f"{sds_gain:>+10.3f}"
    )

# ----------------------------------------------------------------
# Print confusion matrix for Constant model (should be worst)
# and Linear Regression (should be best) in the full ensemble
# ----------------------------------------------------------------

print("\n--- Confusion matrices from full ensemble DS ---")

annotations_full, _, item_names_full, class_names_full, _ = prepare_ds_input(
    models=ordered_models,
    X_test=X_test,
    y_train=y_train,
    n_bins=N_BINS,
)

ds_full = DawidSkeneEM(max_iter=100, tol=1e-6, smoothing=1e-3)
ds_full.fit(annotations_full)

matrices = ds_full.worker_confusion_matrices()

for model_name in ["Constant (training mean)", "Linear regression"]:
    if model_name in matrices:
        print(f"\n{model_name}:")
        print(np.round(matrices[model_name], 3))