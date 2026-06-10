'''
Base implementation of Dawid-Skene EM algorithm
'''

import numpy as np


class DawidSkeneEM:
    def __init__(self, max_iter=100, tol=1e-6, smoothing=1e-3):
        self.max_iter = max_iter
        self.tol = tol
        self.smoothing = smoothing

        self.classes_ = None
        self.items_ = None
        self.workers_ = None

        self.class_prior_ = None
        self.worker_confusions_ = None
        self.posteriors_ = None

    def fit(self, annotations):
        """
        annotations: list of (item_id, worker_id, label)
        """

        self._prepare_data(annotations)

        posteriors = self._initialise_posteriors()

        for _ in range(self.max_iter):
            old_posteriors = posteriors.copy()

            class_prior, worker_confusions = self._m_step(posteriors)
            posteriors = self._e_step(class_prior, worker_confusions)

            diff = np.max(np.abs(posteriors - old_posteriors))
            if diff < self.tol:
                break

        self.class_prior_ = class_prior
        self.worker_confusions_ = worker_confusions
        self.posteriors_ = posteriors

        return self

    def _prepare_data(self, annotations):
        self.items_ = sorted(set(item for item, _, _ in annotations))
        self.workers_ = sorted(set(worker for _, worker, _ in annotations))
        self.classes_ = sorted(set(label for _, _, label in annotations))

        self.item_to_idx_ = {item: i for i, item in enumerate(self.items_)}
        self.worker_to_idx_ = {worker: i for i, worker in enumerate(self.workers_)}
        self.label_to_idx_ = {label: i for i, label in enumerate(self.classes_)}

        self.data_ = [
            (
                self.item_to_idx_[item],
                self.worker_to_idx_[worker],
                self.label_to_idx_[label],
            )
            for item, worker, label in annotations
        ]

    def _initialise_posteriors(self):
        n_items = len(self.items_)
        n_classes = len(self.classes_)

        posteriors = np.ones((n_items, n_classes)) * self.smoothing

        for item_idx, _, observed_label in self.data_:
            posteriors[item_idx, observed_label] += 1

        posteriors /= posteriors.sum(axis=1, keepdims=True)

        return posteriors

    def _m_step(self, posteriors):
        """
        Estimate:
        - class_prior[c] = P(true label = c)
        - worker_confusions[w, c, l] = P(worker w says l | true label c)
        """

        n_workers = len(self.workers_)
        n_classes = len(self.classes_)

        class_prior = posteriors.mean(axis=0)

        worker_confusions = np.ones(
            (n_workers, n_classes, n_classes)
        ) * self.smoothing

        for item_idx, worker_idx, observed_label in self.data_:
            for true_class in range(n_classes):
                worker_confusions[
                    worker_idx,
                    true_class,
                    observed_label
                ] += posteriors[item_idx, true_class]

        worker_confusions /= worker_confusions.sum(axis=2, keepdims=True)

        return class_prior, worker_confusions

    def _e_step(self, class_prior, worker_confusions):
        """
        Estimate:
        posteriors[i, c] = P(true label of item i is c | observed worker labels)
        """

        n_items = len(self.items_)

        log_posteriors = np.log(class_prior + 1e-12)[None, :].repeat(
            n_items,
            axis=0
        )

        for item_idx, worker_idx, observed_label in self.data_:
            log_posteriors[item_idx] += np.log(
                worker_confusions[worker_idx, :, observed_label] + 1e-12
            )

        log_posteriors -= log_posteriors.max(axis=1, keepdims=True)

        posteriors = np.exp(log_posteriors)
        posteriors /= posteriors.sum(axis=1, keepdims=True)

        return posteriors

    def predict(self):
        preds = np.argmax(self.posteriors_, axis=1)

        return {
            item: self.classes_[preds[i]]
            for i, item in enumerate(self.items_)
        }

    def predict_proba(self):
        return {
            item: {
                label: self.posteriors_[i, c]
                for c, label in enumerate(self.classes_)
            }
            for i, item in enumerate(self.items_)
        }

    def worker_confusion_matrices(self):
        return {
            worker: self.worker_confusions_[w]
            for w, worker in enumerate(self.workers_)
        }


if __name__ == "__main__":
    annotations = [
        ("q1", "model_a", "A"),
        ("q1", "model_b", "A"),
        ("q1", "model_c", "B"),
        ("q1", "model_d", "A"),

        ("q2", "model_a", "B"),
        ("q2", "model_b", "B"),
        ("q2", "model_c", "B"),
        ("q2", "model_d", "C"),

        ("q3", "model_a", "A"),
        ("q3", "model_b", "B"),
        ("q3", "model_c", "A"),
        ("q3", "model_d", "A"),

        ("q4", "model_a", "C"),
        ("q4", "model_b", "C"),
        ("q4", "model_c", "A"),
        ("q4", "model_d", "C"),
    ]

    model = DawidSkeneEM()
    model.fit(annotations)

    print(model.predict())
