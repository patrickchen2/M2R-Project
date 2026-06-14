import numpy as np
import torch
import torch.nn.functional as F


class SoftDawidSkene:
    """
    Soft Dawid-Skene for aggregating soft classifier/model outputs.

    Input:
        probs: np.ndarray of shape (n_items, n_models, n_classes)

    Example:
        probs[i, k, l] = probability assigned to class l
                         by model k on question/item i
    """

    def __init__(
        self,
        max_iter=200,
        tol=1e-6,
        polyak_alpha=1e-2,
        m_steps=5,
        lr=1e-4,
        weight_decay=1e-4,
        concentration_scale=20.0,
        eps=1e-8,
        device=None,
        verbose=False,
    ):
        self.max_iter = max_iter
        self.tol = tol

        # Weight of the new E-step posterior estimate.
        # Small values make the EM updates more stable.
        self.polyak_alpha = polyak_alpha

        # Number of inner optimiser steps for the M-step of Pi.
        self.m_steps = m_steps

        self.lr = lr
        self.weight_decay = weight_decay
        self.concentration_scale = concentration_scale
        self.eps = eps
        self.verbose = verbose

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        self.class_prior_ = None
        self.dirichlet_confusions_ = None
        self.confusion_means_ = None
        self.posteriors_ = None

        self.item_names_ = None
        self.model_names_ = None
        self.class_names_ = None

    def fit(
        self,
        probs,
        item_names=None,
        model_names=None,
        class_names=None,
    ):
        """
        probs: array of shape (n_items, n_models, n_classes)

        Each row probs[i, k, :] should be a probability vector.
        """

        probs = np.asarray(probs, dtype=np.float32)

        if probs.ndim != 3:
            raise ValueError("probs must have shape (n_items, n_models, n_classes).")

        n_items, n_models, n_classes = probs.shape

        self.item_names_ = (
            item_names if item_names is not None else list(range(n_items))
        )
        self.model_names_ = (
            model_names if model_names is not None else list(range(n_models))
        )
        self.class_names_ = (
            class_names if class_names is not None else list(range(n_classes))
        )

        probs = self._normalise_probabilities(probs)

        c = torch.tensor(probs, dtype=torch.float32, device=self.device)
        log_c = torch.log(c + self.eps)

        # Initialise latent true-label probabilities using ensemble averaging.
        posteriors = torch.tensor(
            probs.mean(axis=1),
            dtype=torch.float32,
            device=self.device,
        )
        posteriors = posteriors / posteriors.sum(dim=1, keepdim=True)

        # Initialise class prior.
        class_prior = posteriors.mean(dim=0)
        class_prior = class_prior / class_prior.sum()

        # Initialise Dirichlet confusion parameters.
        pi_init = self._initialise_dirichlet_confusions(probs, posteriors.cpu().numpy())

        raw_pi = torch.nn.Parameter(
            self._inv_softplus(
                torch.tensor(
                    pi_init - self.eps,
                    dtype=torch.float32,
                    device=self.device,
                )
            )
        )

        optimiser = torch.optim.AdamW(
            [raw_pi],
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        previous_q = None

        for iteration in range(self.max_iter):
            old_posteriors = posteriors.clone()

            # E-step: update posterior probabilities over true labels.
            with torch.no_grad():
                pi = F.softplus(raw_pi) + self.eps

                new_posteriors = self._raw_e_step(
                    log_c=log_c,
                    class_prior=class_prior,
                    pi=pi,
                )

                posteriors = (
                    (1.0 - self.polyak_alpha) * posteriors
                    + self.polyak_alpha * new_posteriors
                )
                posteriors = posteriors / posteriors.sum(dim=1, keepdim=True)

                # M-step for class prior has a closed form.
                class_prior = posteriors.mean(dim=0)
                class_prior = class_prior / class_prior.sum()

            # M-step for Dirichlet confusion parameters Pi.
            self._optimise_pi(
                optimiser=optimiser,
                raw_pi=raw_pi,
                log_c=log_c,
                posteriors=posteriors.detach(),
            )

            with torch.no_grad():
                pi = F.softplus(raw_pi) + self.eps
                q_value = self._q_function(
                    log_c=log_c,
                    posteriors=posteriors,
                    class_prior=class_prior,
                    pi=pi,
                ).item()

                diff = torch.max(torch.abs(posteriors - old_posteriors)).item()

            if self.verbose:
                print(
                    f"iter={iteration:03d}, "
                    f"diff={diff:.3e}, "
                    f"Q={q_value:.6f}"
                )

            if diff < self.tol:
                break

            if previous_q is not None and abs(q_value - previous_q) < self.tol:
                break

            previous_q = q_value

        with torch.no_grad():
            pi = F.softplus(raw_pi) + self.eps

            self.class_prior_ = class_prior.cpu().numpy()
            self.dirichlet_confusions_ = pi.cpu().numpy()
            self.confusion_means_ = self._dirichlet_mean(
                self.dirichlet_confusions_
            )
            self.posteriors_ = posteriors.cpu().numpy()

        return self

    # ------------------------------------------------------------
    # E-step
    # ------------------------------------------------------------

    def _raw_e_step(self, log_c, class_prior, pi):
        """
        Computes:

            p(t_i = j | C, nu, Pi)

        where each model output c_i^(k) is modelled as:

            c_i^(k) ~ Dirichlet(Pi[k, j, :])

        if the true class is j.
        """

        # pi shape: (n_models, n_classes, n_classes)
        # log_c shape: (n_items, n_models, n_classes)

        log_norm = (
            torch.lgamma(pi.sum(dim=2))
            - torch.lgamma(pi).sum(dim=2)
        )
        # shape: (n_models, n_classes)

        dirichlet_terms = torch.einsum(
            "nkl,kjl->nkj",
            log_c,
            pi - 1.0,
        )
        # shape: (n_items, n_models, n_classes)

        dirichlet_log_probs = dirichlet_terms + log_norm.unsqueeze(0)

        # Sum over models.
        log_likelihood = dirichlet_log_probs.sum(dim=1)
        # shape: (n_items, n_classes)

        log_posterior = torch.log(class_prior + self.eps).unsqueeze(0)
        log_posterior = log_posterior + log_likelihood

        return torch.softmax(log_posterior, dim=1)

    # ------------------------------------------------------------
    # M-step
    # ------------------------------------------------------------

    def _optimise_pi(self, optimiser, raw_pi, log_c, posteriors):
        """
        Optimises the Dirichlet confusion parameters Pi using autodiff.

        This follows the paper's point that the M-step for Pi does not have
        a simple closed-form solution, so we optimise it with AdamW.
        """

        for _ in range(self.m_steps):
            optimiser.zero_grad()

            pi = F.softplus(raw_pi) + self.eps

            q = self._q_function(
                log_c=log_c,
                posteriors=posteriors,
                class_prior=None,
                pi=pi,
                include_class_prior=False,
            )

            loss = -q
            loss.backward()

            torch.nn.utils.clip_grad_norm_([raw_pi], max_norm=10.0)

            optimiser.step()

    def _q_function(
        self,
        log_c,
        posteriors,
        class_prior,
        pi,
        include_class_prior=True,
    ):
        """
        Expected complete-data log likelihood.

        This is the Q function used by EM.
        """

        log_norm = (
            torch.lgamma(pi.sum(dim=2))
            - torch.lgamma(pi).sum(dim=2)
        )
        # shape: (n_models, n_classes)

        dirichlet_terms = torch.einsum(
            "nkl,kjl->nkj",
            log_c,
            pi - 1.0,
        )
        # shape: (n_items, n_models, n_classes)

        dirichlet_log_probs = dirichlet_terms + log_norm.unsqueeze(0)
        log_likelihood = dirichlet_log_probs.sum(dim=1)
        # shape: (n_items, n_classes)

        if include_class_prior:
            log_likelihood = log_likelihood + torch.log(
                class_prior + self.eps
            ).unsqueeze(0)

        q = torch.sum(posteriors * log_likelihood)

        return q / log_c.shape[0]

    # ------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------

    def _initialise_dirichlet_confusions(self, probs, posteriors):
        """
        Initialise Pi using a hard-label Dawid-Skene style estimate.

        We first take argmax labels from each soft prediction, then estimate
        a confusion matrix using the current soft posteriors. Finally we convert
        each row into Dirichlet concentration parameters.
        """

        n_items, n_models, n_classes = probs.shape

        hard_labels = probs.argmax(axis=2)

        counts = np.ones(
            (n_models, n_classes, n_classes),
            dtype=np.float32,
        )

        for i in range(n_items):
            for k in range(n_models):
                observed_label = hard_labels[i, k]

                for true_class in range(n_classes):
                    counts[k, true_class, observed_label] += posteriors[
                        i,
                        true_class,
                    ]

        confusion_probs = counts / counts.sum(axis=2, keepdims=True)

        # SDS Pi is not a row-normalised probability matrix.
        # It is a Dirichlet concentration matrix.
        pi_init = 1.0 + self.concentration_scale * confusion_probs

        return pi_init.astype(np.float32)

    # ------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------

    def _normalise_probabilities(self, probs):
        probs = np.clip(probs, self.eps, 1.0)
        probs = probs / probs.sum(axis=2, keepdims=True)
        return probs

    def _dirichlet_mean(self, pi):
        return pi / pi.sum(axis=2, keepdims=True)

    def _inv_softplus(self, x):
        return torch.log(torch.expm1(torch.clamp(x, min=self.eps)))

    # ------------------------------------------------------------
    # Public output methods
    # ------------------------------------------------------------

    def predict_proba(self):
        """
        Returns aggregated posterior probabilities for each item.
        """

        return {
            self.item_names_[i]: {
                self.class_names_[j]: float(self.posteriors_[i, j])
                for j in range(len(self.class_names_))
            }
            for i in range(len(self.item_names_))
        }

    def predict(self):
        """
        Returns the most likely class for each item.
        """

        preds = self.posteriors_.argmax(axis=1)

        return {
            self.item_names_[i]: self.class_names_[preds[i]]
            for i in range(len(self.item_names_))
        }

    def model_confusion_means(self):
        """
        Returns the mean of each model's Dirichlet confusion distribution.

        Shape conceptually:
            model -> true_class -> predicted_class
        """

        return {
            self.model_names_[k]: self.confusion_means_[k]
            for k in range(len(self.model_names_))
        }

    def dirichlet_confusion_parameters(self):
        """
        Returns raw SDS Pi parameters.

        These are Dirichlet concentration parameters, not ordinary
        row-normalised confusion probabilities.
        """

        return {
            self.model_names_[k]: self.dirichlet_confusions_[k]
            for k in range(len(self.model_names_))
        }


if __name__ == "__main__":
    # Example:
    # 4 questions, 3 models, 3 possible answers/classes A, B, C

    probs = np.array(
        [
            # q1
            [
                [0.80, 0.15, 0.05],  # model_a
                [0.70, 0.20, 0.10],  # model_b
                [0.35, 0.55, 0.10],  # model_c
            ],

            # q2
            [
                [0.10, 0.80, 0.10],
                [0.20, 0.70, 0.10],
                [0.15, 0.75, 0.10],
            ],

            # q3
            [
                [0.45, 0.45, 0.10],
                [0.60, 0.25, 0.15],
                [0.30, 0.60, 0.10],
            ],

            # q4
            [
                [0.05, 0.20, 0.75],
                [0.10, 0.25, 0.65],
                [0.50, 0.20, 0.30],
            ],
        ],
        dtype=np.float32,
    )

    model = SoftDawidSkene(
        max_iter=200,
        polyak_alpha=1e-2,
        m_steps=5,
        lr=1e-4,
        weight_decay=1e-4,
        verbose=True,
    )

    model.fit(
        probs,
        item_names=["q1", "q2", "q3", "q4"],
        model_names=["model_a", "model_b", "model_c"],
        class_names=["A", "B", "C"],
    )

    print("\nPredicted labels:")
    print(model.predict())

    print("\nAggregated probabilities:")
    for item, p in model.predict_proba().items():
        print(item, p)

    print("\nMean confusion matrices:")
    for model_name, matrix in model.model_confusion_means().items():
        print(f"\n{model_name}")
        print(matrix)