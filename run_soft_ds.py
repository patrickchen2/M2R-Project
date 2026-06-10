'''
Runs the Soft Dawid-Skene model on the temperature data by converting data into discrete categories.
'''

from Models import X_train, X_test, y_train, y_test, models
from discretisemodels import prepare_soft_ds_input
from SoftDawidSkene import SoftDawidSkene


probs_for_sds, model_names, item_names, class_names, bin_edges, sigmas = (
    prepare_soft_ds_input(
        models=models,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        n_bins=5,
    )
)

print("Soft Dawid-Skene input shape:")
print(probs_for_sds.shape)

sds = SoftDawidSkene(
    max_iter=200,
    polyak_alpha=1e-2,
    m_steps=5,
    lr=1e-4,
    verbose=True,
)

sds.fit(
    probs_for_sds,
    item_names=item_names,
    model_names=model_names,
    class_names=class_names,
)

print("Predicted temperature bins:")
print(sds.predict())

print("Posterior probabilities:")
for item, probs in sds.predict_proba().items():
    print(item, probs)

print("\n")
print(sds.predict())
for item, probs in list(sds.predict_proba().items())[:20]:
    print(item + ":", list(map(lambda x: round(x, 3), probs.values())))