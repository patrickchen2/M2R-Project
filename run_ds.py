from DawidSkeneEM import DawidSkeneEM
from discretisemodels import prepare_ds_input
from Models import X_train, X_test, y_train, y_test, models

# Train the models
for name, model in models:
    model.fit(X_train, y_train)

# Prepare Dawid-Skene input
annotations, model_names, item_names, class_names, bin_edges = prepare_ds_input(
    models=models,
    X_test=X_test,
    y_train=y_train,
    n_bins=10,
    )

# Run Dawid-Skene
ds = DawidSkeneEM(max_iter=100, tol=1e-6, smoothing=1e-3)
ds.fit(annotations)

# Get predictions
predicted_bins = ds.predict()
posterior_probs = ds.predict_proba()

print("Example annotations:")
for row in annotations[:10]:
    print(row)

print("\nDawid-Skene predictions:")
for item in item_names[:20]:
    print(item, predicted_bins[item])