'''
Classifies different temperature ranges into dicrete bins
'''

import numpy as np
from scipy.special import ndtr
from sklearn.model_selection import train_test_split


def make_temperature_bins(y_train, n_bins=5):
    """
    Turns continuous temperatures into discrete bins.
    Uses simple quantile-based binning

    Example:
        y = 13.2°C
        -> bin_2

    Returns:
        bin_edges: array of bin boundaries
        class_names: readable bin labels
    """

    inner_edges = np.quantile(
        y_train,
        np.linspace(0, 1, n_bins + 1)[1:-1]
    )

    bin_edges = np.concatenate([
        [-np.inf],
        inner_edges,
        [np.inf],
    ])

    class_names = [
        f"bin_{i}: ({bin_edges[i]:.2f}, {bin_edges[i + 1]:.2f}]"
        for i in range(n_bins)
    ]

    return bin_edges, class_names


def estimate_model_sigmas(models, X_calib, y_calib, min_sigma=0.25):
    """
    Estimate each model's uncertainty using a calibration set.

    If a model usually makes large errors, it gets a larger sigma.
    Larger sigma means its probability vector is more spread out.
    """

    sigmas = []

    for name, model in models:
        preds = model.predict(X_calib)
        residuals = y_calib - preds

        sigma = np.sqrt(np.mean(residuals ** 2))
        sigma = max(sigma, min_sigma)

        sigmas.append(sigma)

        print(f"{name}: sigma = {sigma:.3f}")

    return np.array(sigmas)


def regression_models_to_probability_vectors(
    models,
    X,
    sigmas,
    bin_edges,
    eps=1e-8,
):
    """
    Converts regression outputs into Soft Dawid-Skene probability vectors.

    Returns:
        probs with shape:
            (n_items, n_models, n_classes)

    Meaning:
        probs[i, k, j] =
            probability that model k assigns item i to class/bin j
    """

    n_items = len(X)
    n_models = len(models)
    n_classes = len(bin_edges) - 1

    probs = np.zeros(
        (n_items, n_models, n_classes),
        dtype=np.float32,
    )

    for k, (name, model) in enumerate(models):
        mu = model.predict(X)
        sigma = sigmas[k]

        lower = bin_edges[:-1][None, :]
        upper = bin_edges[1:][None, :]

        z_upper = (upper - mu[:, None]) / sigma
        z_lower = (lower - mu[:, None]) / sigma

        probs[:, k, :] = ndtr(z_upper) - ndtr(z_lower)

    probs = np.clip(probs, eps, 1.0)
    probs /= probs.sum(axis=2, keepdims=True)

    return probs


def temperatures_to_bin_labels(temperatures, bin_edges, class_names):
    """
    Convert continuous temperature predictions into discrete bin labels.

    Example:
        13.2 -> "bin_2: (9.20, 12.70]"
    """

    bin_indices = np.digitize(
        temperatures,
        bin_edges[1:-1],
    )

    return [
        class_names[int(idx)]
        for idx in bin_indices
    ]


def prepare_soft_ds_input(
    models,
    X_train,
    y_train,
    X_test,
    n_bins=5,
    calib_size=0.25,
    random_state=68,
):
    """
    Full preparation pipeline.

    This:
        1. splits training data into base-training and calibration
        2. trains the models
        3. creates temperature bins
        4. estimates model uncertainty
        5. creates Soft Dawid-Skene probability vectors
    """

    X_base, X_calib, y_base, y_calib = train_test_split(
        X_train,
        y_train,
        test_size=calib_size,
        random_state=random_state,
        shuffle=True,
    )

    bin_edges, class_names = make_temperature_bins(
        y_train,
        n_bins=n_bins,
    )

    for name, model in models:
        model.fit(X_base, y_base)

    sigmas = estimate_model_sigmas(
        models,
        X_calib,
        y_calib,
    )

    probs_for_sds = regression_models_to_probability_vectors(
        models=models,
        X=X_test,
        sigmas=sigmas,
        bin_edges=bin_edges,
    )

    model_names = [name for name, _ in models]
    item_names = [f"day_{i}" for i in range(len(X_test))]

    return probs_for_sds, model_names, item_names, class_names, bin_edges, sigmas

def prepare_ds_input(models, X_test, y_train, n_bins=5):
    """
    Prepare hard Dawid-Skene annotations from regression models.

    Each regression model predicts a scalar temperature.
    We convert that scalar prediction into a temperature bin label.

    Returns:
        annotations:
            list of (item_name, model_name, predicted_bin_label)

        model_names:
            list of model names

        item_names:
            list of item/day names

        class_names:
            list of bin labels

        bin_edges:
            numeric bin boundaries
    """

    bin_edges, class_names = make_temperature_bins(
        y_train,
        n_bins=n_bins,
    )

    item_names = [
        f"day_{i}"
        for i in range(len(X_test))
    ]

    model_names = [
        name
        for name, _ in models
    ]

    annotations = []

    for model_name, model in models:
        temperature_predictions = model.predict(X_test)

        predicted_labels = temperatures_to_bin_labels(
            temperature_predictions,
            bin_edges,
            class_names,
        )

        for item_name, predicted_label in zip(item_names, predicted_labels):
            annotations.append(
                (
                    item_name,
                    model_name,
                    predicted_label,
                )
            )

    return annotations, model_names, item_names, class_names, bin_edges