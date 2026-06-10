"""
Dawid-Skene EM with spectral initialisation.

Based on:
Zhang, Chen, Zhou, Jordan.
"Spectral Methods Meet EM: A Provably Optimal Algorithm for Crowdsourcing."
NeurIPS 2014.

Input annotations:

    (item_id, worker_id, label)

Example:

    ("day_0", "Linear regression", "bin_3")
"""

import numpy as np


class DawidSkeneEM:
    def __init__(
        self,
        max_iter=100,
        tol=1e-6,
        smoothing=1e-3,
        init="spectral",
        class_names=None,
        random_state=0,
        spectral_restarts=20,
        spectral_power_iters=100,
        inverse_condition_threshold=1e10,
        symmetrise_tensor=True,
        use_uniform_prior=True,
        verbose=False,
    ):
        """
        Parameters
        ----------
        max_iter:
            Maximum number of EM iterations.

        tol:
            Convergence tolerance based on max posterior change.

        smoothing:
            Small positive value used when projecting matrices to the simplex.
            This is a numerical stabiliser.

        init:
            "spectral" or "majority".

        class_names:
            Full list of possible labels/classes.
            For your weather project, pass all temperature-bin labels here.
            This prevents missing bins from disappearing.

        random_state:
            Seed used when partitioning workers and in tensor power restarts.

        spectral_restarts:
            Number of random restarts in the tensor power method.

        spectral_power_iters:
            Number of power iterations per restart.

        inverse_condition_threshold:
            If a matrix has condition number above this threshold, use
            pseudoinverse instead of inverse.

        symmetrise_tensor:
            Whether to symmetrise the whitened empirical tensor before tensor
            decomposition. The population tensor is symmetric; empirical noise
            can make it slightly asymmetric.

        use_uniform_prior:
            If True, the EM E-step uses a uniform prior over true labels,
            matching the paper's Stage 2 presentation.

        verbose:
            If True, prints convergence diagnostics.
        """

        self.max_iter = max_iter
        self.tol = tol
        self.smoothing = smoothing
        self.init = init
        self.class_names = class_names
        self.random_state = random_state
        self.spectral_restarts = spectral_restarts
        self.spectral_power_iters = spectral_power_iters
        self.inverse_condition_threshold = inverse_condition_threshold
        self.symmetrise_tensor = symmetrise_tensor
        self.use_uniform_prior = use_uniform_prior
        self.verbose = verbose

        self.items_ = None
        self.workers_ = None
        self.classes_ = None

        self.class_prior_ = None
        self.spectral_class_weights_ = None
        self.worker_confusions_ = None
        self.posteriors_ = None

    # ------------------------------------------------------------------
    # Main fitting method
    # ------------------------------------------------------------------

    def fit(self, annotations):
        """
        Fit Dawid-Skene using either spectral or majority initialisation.

        annotations:
            list of (item_id, worker_id, label)
        """

        self._prepare_data(annotations)

        if self.init == "spectral":
            try:
                spectral_weights, worker_confusions = self._spectral_initialise()

                self.spectral_class_weights_ = spectral_weights

                posteriors = self._e_step(worker_confusions)

            except Exception as e:
                print("Spectral initialisation failed; falling back to majority vote.")
                print("Reason:", e)

                posteriors = self._initialise_posteriors_majority_vote()
                worker_confusions = self._m_step(posteriors)

        elif self.init == "majority":
            posteriors = self._initialise_posteriors_majority_vote()
            worker_confusions = self._m_step(posteriors)

        else:
            raise ValueError("init must be either 'spectral' or 'majority'")

        for iteration in range(self.max_iter):
            old_posteriors = posteriors.copy()

            worker_confusions = self._m_step(posteriors)
            posteriors = self._e_step(worker_confusions)

            diff = np.max(np.abs(posteriors - old_posteriors))

            if self.verbose:
                print(
                    f"iteration={iteration:03d}, "
                    f"posterior_change={diff:.6e}, "
                    f"log_likelihood={self._observed_log_likelihood_from_params(worker_confusions):.6f}"
                )

            if diff < self.tol:
                break

        self.worker_confusions_ = worker_confusions
        self.posteriors_ = posteriors

        if self.use_uniform_prior:
            self.class_prior_ = np.ones(self.n_classes_) / self.n_classes_
        else:
            self.class_prior_ = posteriors.mean(axis=0)
            self.class_prior_ = self._normalise_vector(self.class_prior_)

        return self

    # ------------------------------------------------------------------
    # Data preparation
    # ------------------------------------------------------------------

    def _prepare_data(self, annotations):
        self.items_ = sorted(set(item for item, _, _ in annotations))
        self.workers_ = sorted(set(worker for _, worker, _ in annotations))

        if self.class_names is None:
            self.classes_ = sorted(set(label for _, _, label in annotations))
        else:
            self.classes_ = list(self.class_names)

        self.item_to_idx_ = {
            item: i for i, item in enumerate(self.items_)
        }

        self.worker_to_idx_ = {
            worker: i for i, worker in enumerate(self.workers_)
        }

        self.label_to_idx_ = {
            label: i for i, label in enumerate(self.classes_)
        }

        self.data_ = []

        for item, worker, label in annotations:
            if label not in self.label_to_idx_:
                raise ValueError(
                    f"Label {label!r} was not found in class_names. "
                    f"Known classes are: {self.classes_}"
                )

            self.data_.append(
                (
                    self.item_to_idx_[item],
                    self.worker_to_idx_[worker],
                    self.label_to_idx_[label],
                )
            )

        self.n_items_ = len(self.items_)
        self.n_workers_ = len(self.workers_)
        self.n_classes_ = len(self.classes_)

        # label_tensor[worker, item, observed_label] = 1
        self.label_tensor_ = np.zeros(
            (self.n_workers_, self.n_items_, self.n_classes_),
            dtype=float,
        )

        for item_idx, worker_idx, observed_label in self.data_:
            self.label_tensor_[worker_idx, item_idx, observed_label] = 1.0

    # ------------------------------------------------------------------
    # Majority-vote initialisation
    # ------------------------------------------------------------------

    def _initialise_posteriors_majority_vote(self):
        posteriors = np.ones(
            (self.n_items_, self.n_classes_),
            dtype=float,
        ) * self.smoothing

        for item_idx, _, observed_label in self.data_:
            posteriors[item_idx, observed_label] += 1.0

        posteriors /= posteriors.sum(axis=1, keepdims=True)

        return posteriors

    # ------------------------------------------------------------------
    # EM algorithm in paper orientation
    # ------------------------------------------------------------------

    def _m_step(self, posteriors):
        """
        M-step.

        Estimate worker confusion matrices using soft counts.

        Paper orientation:

            worker_confusions[worker, observed_label, true_label]

        Therefore, for each worker and true label, the probabilities over
        observed labels sum to 1.
        """

        k = self.n_classes_

        worker_confusions = np.ones(
            (self.n_workers_, k, k),
            dtype=float,
        ) * self.smoothing

        for item_idx, worker_idx, observed_label in self.data_:
            for true_label in range(k):
                worker_confusions[
                    worker_idx,
                    observed_label,
                    true_label,
                ] += posteriors[item_idx, true_label]

        worker_confusions = self._normalise_worker_columns(worker_confusions)

        return worker_confusions

    def _e_step(self, worker_confusions):
        """
        E-step.

        Estimate:

            posteriors[item, true_label]
                = P(true label | observed worker labels)

        If use_uniform_prior=True, this matches the paper's Stage 2 EM
        formulation, where the E-step is proportional to the product of
        worker confusion probabilities.
        """

        log_posteriors = np.zeros(
            (self.n_items_, self.n_classes_),
            dtype=float,
        )

        if not self.use_uniform_prior:
            if self.class_prior_ is None:
                prior = np.ones(self.n_classes_) / self.n_classes_
            else:
                prior = self.class_prior_

            log_posteriors += np.log(prior + 1e-12)[None, :]

        for item_idx, worker_idx, observed_label in self.data_:
            log_posteriors[item_idx] += np.log(
                worker_confusions[
                    worker_idx,
                    observed_label,
                    :,
                ] + 1e-12
            )

        log_posteriors -= log_posteriors.max(axis=1, keepdims=True)

        posteriors = np.exp(log_posteriors)
        posteriors /= posteriors.sum(axis=1, keepdims=True)

        return posteriors

    # ------------------------------------------------------------------
    # Spectral initialisation
    # ------------------------------------------------------------------

    def _spectral_initialise(self):
        """
        Spectral initialisation following the paper's high-level Algorithm 1.

        Steps:
            1. Partition workers into 3 groups.
            2. Compute group-aggregated labels.
            3. Estimate group-level confusion matrices using moments.
            4. Estimate individual worker confusion matrices.
            5. Return spectral class weights and worker confusion matrices.
        """

        if self.n_workers_ < 3:
            raise ValueError("Spectral initialisation requires at least 3 workers.")

        groups = self._partition_workers_into_three_groups()

        Z = self._group_aggregated_labels(groups)

        group_confusions, prior_estimates = self._estimate_group_confusions(Z)

        spectral_weights = self._normalise_vector(
            np.mean(prior_estimates, axis=0)
        )

        worker_confusions = self._estimate_individual_worker_confusions(
            groups=groups,
            Z=Z,
            group_confusions=group_confusions,
            class_weights=spectral_weights,
        )

        worker_confusions = self._normalise_worker_columns(
            worker_confusions,
            eps=self.smoothing,
        )

        if self.verbose:
            self.spectral_diagnostics(
                groups=groups,
                group_confusions=group_confusions,
                class_weights=spectral_weights,
            )

        return spectral_weights, worker_confusions

    def _partition_workers_into_three_groups(self):
        rng = np.random.default_rng(self.random_state)

        worker_indices = np.arange(self.n_workers_)
        rng.shuffle(worker_indices)

        groups = np.array_split(worker_indices, 3)

        if any(len(group) == 0 for group in groups):
            raise ValueError("Each spectral group must contain at least one worker.")

        return [np.array(group, dtype=int) for group in groups]

    def _group_aggregated_labels(self, groups):
        """
        Compute group-aggregated label vectors.

        Z[g, item, observed_label] is the average label vector for group g.
        """

        Z = np.zeros(
            (3, self.n_items_, self.n_classes_),
            dtype=float,
        )

        for g_idx, group in enumerate(groups):
            Z[g_idx] = self.label_tensor_[group].mean(axis=0)

        return Z

    def _estimate_group_confusions(self, Z):
        """
        Estimate group-level confusion matrices.

        Returns
        -------
        group_confusions:
            shape = (3, n_classes, n_classes)

            group_confusions[g, observed_label, true_label]

        prior_estimates:
            one estimated class-weight vector per group.
        """

        k = self.n_classes_

        group_confusions = np.zeros(
            (3, k, k),
            dtype=float,
        )

        prior_estimates = []

        # Python indices 0, 1, 2 correspond to paper groups 1, 2, 3.
        #
        # These permutations estimate C_c using the other two groups.
        permutations = [
            (1, 2, 0),
            (2, 0, 1),
            (0, 1, 2),
        ]

        for a, b, c in permutations:
            M_cb = self._mean_outer(Z[c], Z[b])
            M_ab = self._mean_outer(Z[a], Z[b])

            M_ca = self._mean_outer(Z[c], Z[a])
            M_ba = self._mean_outer(Z[b], Z[a])

            Z0_a = (M_cb @ self._safe_inverse(M_ab) @ Z[a].T).T
            Z0_b = (M_ca @ self._safe_inverse(M_ba) @ Z[b].T).T

            M2 = self._mean_outer(Z0_a, Z0_b)

            M3 = np.einsum(
                "ni,nj,nl->ijl",
                Z0_a,
                Z0_b,
                Z[c],
            ) / self.n_items_

            C_c, w_c = self._tensor_decompose_moments(M2, M3)

            C_c = self._normalise_columns(C_c, eps=self.smoothing)

            group_confusions[c] = C_c
            prior_estimates.append(w_c)

        return group_confusions, np.array(prior_estimates)

    def _estimate_individual_worker_confusions(
        self,
        groups,
        Z,
        group_confusions,
        class_weights,
    ):
        """
        Estimate individual worker confusion matrices.

        Paper-style formula:

            C_i = normalize{ E[z_i Z_a^T] (W C_a^T)^(-1) }

        where:
            C_i has shape observed_label x true_label.
        """

        k = self.n_classes_

        worker_confusions = np.zeros(
            (self.n_workers_, k, k),
            dtype=float,
        )

        W = np.diag(class_weights)

        for g_idx, group in enumerate(groups):
            # Choose a reference group different from g_idx.
            reference_group = (g_idx + 1) % 3

            C_a = group_confusions[reference_group]

            denominator = W @ C_a.T
            denominator_inv = self._safe_inverse(denominator)

            for worker_idx in group:
                z_i = self.label_tensor_[worker_idx]

                empirical_moment = self._mean_outer(
                    z_i,
                    Z[reference_group],
                )

                C_i = empirical_moment @ denominator_inv

                C_i = self._normalise_columns(C_i, eps=self.smoothing)

                worker_confusions[worker_idx] = C_i

        return worker_confusions

    # ------------------------------------------------------------------
    # Tensor decomposition
    # ------------------------------------------------------------------

    def _tensor_decompose_moments(self, M2, M3):
        """
        Decompose M2 and M3 using whitening and tensor power method.

        Returns
        -------
        C_group:
            shape = observed_label x true_label

        w_group:
            estimated class weights.
        """

        k = self.n_classes_

        # Empirical M2 may not be perfectly symmetric.
        M2 = 0.5 * (M2 + M2.T)

        U, S, _ = np.linalg.svd(M2)

        U = U[:, :k]
        S = S[:k]

        if np.any(S <= 1e-12):
            raise ValueError(
                "M2 has non-positive or near-zero singular values; "
                "whitening failed."
            )

        Q = U @ np.diag(1.0 / np.sqrt(S))

        whitened_tensor = np.einsum(
            "abc,ap,bq,cr->pqr",
            M3,
            Q,
            Q,
            Q,
        )

        if self.symmetrise_tensor:
            whitened_tensor = self._symmetrise_tensor(whitened_tensor)

        eigenvalues, eigenvectors = self._robust_tensor_power_method(
            whitened_tensor,
            n_components=k,
        )

        recovered_columns = []
        recovered_weights = []

        for alpha, v in zip(eigenvalues, eigenvectors):
            alpha = abs(alpha)

            if alpha <= 1e-12:
                raise ValueError("Tensor component had near-zero eigenvalue.")

            # Paper relationship after whitening:
            #     alpha_h ≈ 1 / sqrt(w_h)
            w = 1.0 / (alpha ** 2)

            # Undo whitening:
            #     mu_h = (Q^T)^(-1) alpha_h v_h
            mu = np.linalg.solve(Q.T, alpha * v)

            recovered_columns.append(mu)
            recovered_weights.append(w)

        recovered_columns = np.column_stack(recovered_columns)
        recovered_weights = np.array(recovered_weights)

        C_group, w_group = self._align_recovered_columns(
            recovered_columns,
            recovered_weights,
        )

        C_group = self._normalise_columns(C_group, eps=self.smoothing)
        w_group = self._normalise_vector(w_group)

        return C_group, w_group

    def _robust_tensor_power_method(self, tensor, n_components):
        """
        Restarted tensor power method with deflation.

        This captures the role of the robust tensor power method in the paper,
        although it is still a practical simplified implementation.
        """

        rng = np.random.default_rng(self.random_state)

        k = tensor.shape[0]
        T = tensor.copy()

        eigenvalues = []
        eigenvectors = []

        for _ in range(n_components):
            best_lambda = None
            best_vector = None

            for _ in range(self.spectral_restarts):
                v = rng.normal(size=k)
                v /= np.linalg.norm(v) + 1e-12

                for _ in range(self.spectral_power_iters):
                    v_new = np.einsum(
                        "abc,b,c->a",
                        T,
                        v,
                        v,
                    )

                    norm = np.linalg.norm(v_new)

                    if norm < 1e-12:
                        break

                    v = v_new / norm

                lam = np.einsum(
                    "abc,a,b,c->",
                    T,
                    v,
                    v,
                    v,
                )

                if best_lambda is None or abs(lam) > abs(best_lambda):
                    best_lambda = lam
                    best_vector = v.copy()

            if best_lambda is None or best_vector is None:
                raise RuntimeError("Tensor power method failed.")

            if best_lambda < 0:
                best_lambda = -best_lambda
                best_vector = -best_vector

            eigenvalues.append(best_lambda)
            eigenvectors.append(best_vector)

            T -= best_lambda * np.einsum(
                "a,b,c->abc",
                best_vector,
                best_vector,
                best_vector,
            )

        return np.array(eigenvalues), np.array(eigenvectors)

    def _align_recovered_columns(self, recovered_columns, recovered_weights):
        """
        Tensor decomposition recovers columns up to permutation.

        Under the paper's diagonal dominance assumption, a recovered column is
        assigned to the class corresponding to its largest coordinate.
        """

        k = self.n_classes_

        C = np.zeros((k, k), dtype=float)
        w = np.zeros(k, dtype=float)

        unused = set(range(k))

        for true_label in range(k):
            if unused:
                chosen = max(
                    unused,
                    key=lambda h: recovered_columns[true_label, h],
                )
                unused.remove(chosen)
            else:
                chosen = int(np.argmax(recovered_columns[true_label]))

            C[:, true_label] = recovered_columns[:, chosen]
            w[true_label] = recovered_weights[chosen]

        return C, w

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def spectral_diagnostics(self, groups, group_confusions, class_weights):
        print("\nSpectral diagnostics")
        print("--------------------")
        print("Class weights:", class_weights)
        print("Minimum class weight:", class_weights.min())

        for g_idx, C in enumerate(group_confusions):
            print(f"\nGroup {g_idx}")
            print("Workers:", [self.workers_[w] for w in groups[g_idx]])
            print("Rank:", np.linalg.matrix_rank(C))
            print("Condition number:", np.linalg.cond(C))

            diagonal_gaps = []

            for true_label in range(self.n_classes_):
                correct_prob = C[true_label, true_label]

                incorrect = [
                    C[observed_label, true_label]
                    for observed_label in range(self.n_classes_)
                    if observed_label != true_label
                ]

                if incorrect:
                    diagonal_gaps.append(correct_prob - max(incorrect))
                else:
                    diagonal_gaps.append(correct_prob)

            print("Minimum diagonal gap:", min(diagonal_gaps))

    # ------------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------------

    def _mean_outer(self, A, B):
        """
        Empirical mean of outer products:

            (1/n) sum_i A_i B_i^T
        """

        return np.einsum("ni,nj->ij", A, B) / A.shape[0]

    def _safe_inverse(self, A):
        """
        Use ordinary inverse when the matrix is well-conditioned.
        Fall back to pseudoinverse otherwise.
        """

        cond = np.linalg.cond(A)

        if np.isfinite(cond) and cond < self.inverse_condition_threshold:
            return np.linalg.inv(A)

        return np.linalg.pinv(A)

    def _normalise_vector(self, vector):
        vector = np.maximum(vector, self.smoothing)
        return vector / vector.sum()

    def _normalise_columns(self, matrix, eps=0.0):
        """
        Normalise columns of a matrix so each column sums to 1.

        Used for paper-orientation confusion matrices:

            C[observed_label, true_label]
        """

        matrix = np.maximum(matrix, eps)

        denom = matrix.sum(axis=0, keepdims=True)
        denom = np.where(denom <= 0, 1.0, denom)

        return matrix / denom

    def _normalise_worker_columns(self, worker_confusions, eps=0.0):
        """
        Normalise each worker confusion matrix over observed labels.

        worker_confusions shape:
            worker x observed_label x true_label
        """

        worker_confusions = np.maximum(worker_confusions, eps)

        denom = worker_confusions.sum(axis=1, keepdims=True)
        denom = np.where(denom <= 0, 1.0, denom)

        return worker_confusions / denom

    def _symmetrise_tensor(self, T):
        return (
            T
            + T.transpose(0, 2, 1)
            + T.transpose(1, 0, 2)
            + T.transpose(1, 2, 0)
            + T.transpose(2, 0, 1)
            + T.transpose(2, 1, 0)
        ) / 6.0

    # ------------------------------------------------------------------
    # Likelihood and outputs
    # ------------------------------------------------------------------

    def _observed_log_likelihood_from_params(self, worker_confusions):
        """
        Observed log likelihood under current worker confusions.

        Uses uniform class prior if use_uniform_prior=True.
        """

        item_to_labels = [[] for _ in range(self.n_items_)]

        for item_idx, worker_idx, observed_label in self.data_:
            item_to_labels[item_idx].append((worker_idx, observed_label))

        total = 0.0

        for item_idx in range(self.n_items_):
            log_scores = np.zeros(self.n_classes_)

            if not self.use_uniform_prior:
                if self.class_prior_ is None:
                    prior = np.ones(self.n_classes_) / self.n_classes_
                else:
                    prior = self.class_prior_

                log_scores += np.log(prior + 1e-12)

            for worker_idx, observed_label in item_to_labels[item_idx]:
                log_scores += np.log(
                    worker_confusions[worker_idx, observed_label, :] + 1e-12
                )

            max_score = np.max(log_scores)

            total += max_score + np.log(
                np.sum(np.exp(log_scores - max_score))
            )

        return total

    def observed_log_likelihood(self):
        if self.worker_confusions_ is None:
            raise RuntimeError("Model has not been fitted yet.")

        return self._observed_log_likelihood_from_params(
            self.worker_confusions_
        )

    def predict(self):
        preds = np.argmax(self.posteriors_, axis=1)

        return {
            item: self.classes_[preds[i]]
            for i, item in enumerate(self.items_)
        }

    def predict_proba(self):
        return {
            item: {
                label: float(self.posteriors_[i, c])
                for c, label in enumerate(self.classes_)
            }
            for i, item in enumerate(self.items_)
        }

    def worker_confusion_matrices(self):
        """
        Returns confusion matrices in paper orientation:

            matrix[observed_label, true_label]

        Therefore columns sum to 1.
        """

        return {
            worker: self.worker_confusions_[w]
            for w, worker in enumerate(self.workers_)
        }

    def worker_confusion_matrices_em_orientation(self):
        """
        Optional helper if you want the old orientation:

            matrix[true_label, observed_label]
        """

        return {
            worker: self.worker_confusions_[w].T
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

    model = DawidSkeneEM(
        init="spectral",
        class_names=["A", "B", "C"],
        max_iter=100,
        tol=1e-6,
        smoothing=1e-3,
        random_state=42,
        verbose=True,
    )

    model.fit(annotations)

    print("\nPredicted true answers:")
    print(model.predict())

    print("\nPosterior probabilities:")
    for question, probs in model.predict_proba().items():
        print(question, probs)

    print("\nObserved log likelihood:")
    print(model.observed_log_likelihood())

    print("\nEstimated model confusion matrices")
    print("Orientation: matrix[observed_label, true_label]")
    for worker, matrix in model.worker_confusion_matrices().items():
        print(f"\n{worker}")
        print(matrix)