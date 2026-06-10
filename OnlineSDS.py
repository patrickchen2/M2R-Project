"""
online_sds.py

Online Soft Dawid-Skene, based on Appendix F of Kuzin et al.

The idea:
    1. Run full SDS on a small learning batch to estimate confusion matrices.
    2. Fix those confusion matrices.
    3. For each remaining test point, run only the E-step to get a prediction.

Why bother:
    Full SDS needs all the data at once (batch). The online version lets you
    make predictions on new data without re-running the full algorithm each
    time, which is much faster.

The trade-off:
    Slightly worse calibration than full SDS (as shown in the paper's Figure 7)
    but a fraction of the compute cost.

Usage:
    python online_sds.py
"""

import numpy as np
import torch
from sklearn.metrics import mean_absolute_error

from Models import X_train, X_test, y_train, y_test, models
from discretisemodels import prepare_soft_ds_input
from SoftDawidSkene import SoftDawidSkene

for name, model in models:
    model.fit(X_train, y_train)


# --- Helper ---

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


# --- Split test set into learning batch and inference batch ---
# The paper uses 10000/40000 on ImageNet. We scale proportionally.

N_BINS = 5
n_test = len(X_test)
n_learn = max(10, n_test // 5)   # use 20% of test set to learn confusion matrices

X_learn, X_infer = X_test[:n_learn], X_test[n_learn:]
y_learn, y_infer = y_test[:n_learn], y_test[n_learn:]

print(f"Test set split: {n_learn} learning points, {len(X_infer)} inference points")


# --- Step 1: run full SDS on the learning batch to get confusion matrices ---

probs_learn, model_names, item_names_learn, class_names, bin_edges, sigmas = (
    prepare_soft_ds_input(
        models=models, X_train=X_train, y_train=y_train, X_test=X_learn, n_bins=N_BINS
    )
)
midpoints = bin_midpoints(bin_edges)

sds = SoftDawidSkene(max_iter=200, polyak_alpha=1e-2, m_steps=5, lr=1e-4, verbose=False)
sds.fit(probs_learn, item_names=item_names_learn, model_names=model_names, class_names=class_names)

# The learned confusion matrices and class prior are now fixed
pi_fixed = sds.dirichlet_confusions_    # shape (n_models, n_classes, n_classes)
nu_fixed = sds.class_prior_             # shape (n_classes,)


# --- Step 2: E-step only for each inference point ---
# This is the online part. For each new point we only run the E-step,
# using the fixed confusion matrices from above.
# The E-step formula is equation (3) from the paper:
#
#   log p(t_i = j | c_i, nu, Pi) ∝ log nu_j
#       + sum_k sum_l (pi_jl - 1) * log c_il
#       - sum_k [ sum_l log Gamma(pi_jl) - log Gamma(sum_l pi_jl) ]

def e_step_single_point(probs_single, pi, nu, eps=1e-8):
    """
    One E-step for a single data point.

    probs_single: shape (n_models, n_classes) -- soft predictions from each model
    pi:           shape (n_models, n_classes, n_classes) -- fixed Dirichlet params
    nu:           shape (n_classes,) -- fixed class prior

    Returns: shape (n_classes,) -- posterior over true class
    """
    c = torch.tensor(probs_single, dtype=torch.float32)
    c = torch.clamp(c, eps, 1.0)
    log_c = torch.log(c)

    pi_t = torch.tensor(pi, dtype=torch.float32)
    nu_t = torch.tensor(nu, dtype=torch.float32)

    # Log normaliser term for each model and true class
    log_norm = torch.lgamma(pi_t.sum(dim=2)) - torch.lgamma(pi_t).sum(dim=2)
    # shape: (n_models, n_classes)

    # Dirichlet log-likelihood term
    dirichlet_terms = torch.einsum("kl,kjl->kj", log_c, pi_t - 1.0)
    # shape: (n_models, n_classes)

    log_likelihood = (dirichlet_terms + log_norm).sum(dim=0)
    # shape: (n_classes,)

    log_posterior = torch.log(nu_t + eps) + log_likelihood
    posterior = torch.softmax(log_posterior, dim=0)

    return posterior.numpy()


# Get soft probability vectors for the inference batch
probs_infer, _, item_names_infer, _, _, _ = prepare_soft_ds_input(
    models=models, X_train=X_train, y_train=y_train, X_test=X_infer, n_bins=N_BINS
)

# Run E-step on each inference point independently
online_preds = {}
for i, item in enumerate(item_names_infer):
    posterior = e_step_single_point(probs_infer[i], pi_fixed, nu_fixed)
    predicted_class = class_names[int(np.argmax(posterior))]
    online_preds[item] = predicted_class


# --- Compare online SDS vs full SDS vs ensemble average on inference batch ---

# Full SDS on the inference batch (for comparison)
sds_full = SoftDawidSkene(max_iter=200, polyak_alpha=1e-2, m_steps=5, lr=1e-4, verbose=False)
sds_full.fit(probs_infer, item_names=item_names_infer, model_names=model_names, class_names=class_names)

# Ensemble average on inference batch
ensemble_preds = np.mean([m.predict(X_infer) for _, m in models], axis=0)

# Convert to temperatures
online_temps = preds_to_temps(online_preds,       item_names_infer, class_names, midpoints)
full_temps   = preds_to_temps(sds_full.predict(), item_names_infer, class_names, midpoints)

print()
print("MAE on inference batch (degrees C, lower is better)")
print("-" * 45)
print(f"  {'Ensemble average':<28} {mean_absolute_error(y_infer, ensemble_preds):.3f}")
print(f"  {'Online SDS':<28} {mean_absolute_error(y_infer, online_temps):.3f}")
print(f"  {'Full SDS':<28} {mean_absolute_error(y_infer, full_temps):.3f}")
print("-" * 45)
print()
print("Online SDS learned its confusion matrices from only")
print(f"{n_learn} points, then predicted the remaining {len(X_infer)} individually.")
print("Full SDS used all", len(X_infer), "inference points at once.")