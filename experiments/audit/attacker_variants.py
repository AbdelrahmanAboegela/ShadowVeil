"""Nonlinear and sparse-dictionary attacker variants for the shadow-revocation protocol.

The primary confirmation protocol (stage10/stage11) uses a linear RidgeClassifier
on sampled SSM coordinates. This module adds two stronger, still-adaptive
attackers so shadow-condition accuracy is checked against more than a linear
probe:

- ``fit_predict_mlp``: a small nonlinear classifier on the same raw features.
- ``fit_predict_sae``: a sparse autoencoder trained on defender-only
  (untreated-condition) cache states, then a linear probe on the resulting
  sparse code. This mirrors the SAE-for-SSM-state literature (WriteSAE,
  activation-subspace-bottleneck steering) applied here as an attacker rather
  than a steering mechanism; see sae_literature_review.md.

Both functions share the ``fit_predict`` signature used by stage10/stage11 so
they can be dropped into the same evaluation loop.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def wilson(successes, total, z=1.959963984540054):
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return [float(center - half), float(center + half)]


def _score(y_test, prediction):
    successes = int((prediction == y_test).sum())
    return {
        "accuracy": float(accuracy_score(y_test, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, prediction)),
        "wilson_95": wilson(successes, len(y_test)),
        "correct": (prediction == y_test).astype(int).tolist(),
        "predictions": prediction.astype(int).tolist(),
    }


def fit_predict_linear(x_train, y_train, x_test, y_test):
    probe = make_pipeline(StandardScaler(), RidgeClassifier(alpha=10.0))
    probe.fit(x_train, y_train)
    return _score(y_test, probe.predict(x_test))


def fit_predict_mlp(x_train, y_train, x_test, y_test, seed=0):
    probe = make_pipeline(
        StandardScaler(),
        MLPClassifier(
            hidden_layer_sizes=(256,),
            alpha=1e-3,
            max_iter=400,
            random_state=seed,
            early_stopping=True,
            n_iter_no_change=15,
        ),
    )
    probe.fit(x_train, y_train)
    return _score(y_test, probe.predict(x_test))


class SparseAutoencoder(nn.Module):
    """A single-hidden-layer overcomplete SAE with an L1 sparsity penalty.

    Standard tied-decoder-free dictionary-learning SAE, matching the setup
    used in the transformer SAE literature this borrows from: ReLU encoder
    into an overcomplete latent, linear decoder, L1 penalty on the latent
    activations to keep it sparse.

    Note (kept for anyone extending this): this L1-penalty formulation does
    not reliably reach a strict sparsity target on this data -- an L1-weight
    search up to 300x the default here still left ~45% of units active per
    example, and a TopK variant was prototyped as an alternative but is not
    what produced the SAE-attacker numbers reported in the paper. This class
    is the one actually used for those numbers; do not swap it for a
    stricter-sparsity variant without re-running and re-verifying stage19.
    """

    def __init__(self, input_dim: int, latent_dim: int):
        super().__init__()
        self.encoder = nn.Linear(input_dim, latent_dim)
        self.decoder = nn.Linear(latent_dim, input_dim, bias=False)

    def forward(self, x):
        latent = torch.relu(self.encoder(x))
        reconstruction = self.decoder(latent)
        return reconstruction, latent


def train_sae(
    x_defender: np.ndarray,
    latent_multiplier: int = 2,
    max_latent_dim: int = 1024,
    l1_weight: float = 1e-3,
    epochs: int = 80,
    lr: float = 3e-3,
    seed: int = 0,
    device: str = "cpu",
):
    """Trains an SAE on defender-only cache features and returns (sae, mean, std).

    ``x_defender`` must come from a split disjoint from the attacker/test
    documents used downstream, matching the discipline in
    stage3_feature_scrubber.py (learn on a defender-only split, evaluate on a
    disjoint attacker/test split).

    ``max_latent_dim`` caps the overcomplete dictionary size regardless of
    ``latent_multiplier``. Our sampled-coordinate features are 24,576-wide
    (1024 coordinates x 24 layers); an 8x-overcomplete SAE at that width would
    need a 24576x196608 encoder -- about 19GB for that one weight matrix, ~38GB
    for encoder+decoder, and several times that again for Adam's optimizer
    state. The cap keeps this tractable on a consumer machine and keeps the
    latent count sane relative to the ~100-example defender splits used here
    (an overcomplete dictionary is normal in the SAE literature relative to
    input width, but a six-figure latent count fit on ~100 examples is not a
    meaningful dictionary regardless of memory).
    """
    torch.manual_seed(seed)
    mean = x_defender.mean(axis=0, keepdims=True)
    std = x_defender.std(axis=0, keepdims=True) + 1e-6
    normalized = (x_defender - mean) / std
    tensor = torch.tensor(normalized, dtype=torch.float32, device=device)

    input_dim = x_defender.shape[1]
    latent_dim = min(input_dim * latent_multiplier, max_latent_dim)
    sae = SparseAutoencoder(input_dim, latent_dim).to(device)
    optimizer = torch.optim.Adam(sae.parameters(), lr=lr)

    sae.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        reconstruction, latent = sae(tensor)
        reconstruction_loss = torch.mean((reconstruction - tensor) ** 2)
        sparsity_loss = torch.mean(torch.abs(latent))
        loss = reconstruction_loss + l1_weight * sparsity_loss
        loss.backward()
        optimizer.step()
    sae.eval()
    with torch.no_grad():
        _, latent = sae(tensor)
        achieved_l0 = float((latent > 0).float().sum(dim=1).mean())
    sae.chosen_l1_weight = l1_weight
    sae.achieved_l0 = achieved_l0
    sae.target_l0 = None
    return sae, mean, std


def sae_encode(sae: SparseAutoencoder, x: np.ndarray, mean, std, device: str = "cpu") -> np.ndarray:
    normalized = (x - mean) / std
    tensor = torch.tensor(normalized, dtype=torch.float32, device=device)
    with torch.no_grad():
        _, latent = sae(tensor)
    return latent.cpu().numpy()


def fit_predict_sae(
    sae_bundle,
    x_defender_for_sae: np.ndarray,
    x_train: np.ndarray,
    y_train,
    x_test: np.ndarray,
    y_test,
    device: str = "cpu",
):
    """Fits a linear probe on sparse codes from an already-trained SAE.

    ``sae_bundle`` is the (sae, mean, std) tuple from ``train_sae``, trained
    once on the defender-only split and reused across every condition --
    training a fresh SAE per condition would be redundant, since the SAE only
    depends on the defender split, not on which condition is being scored.
    ``x_defender_for_sae`` is passed again here only to compute the
    reconstruction-MSE diagnostic against the same normalization.

    Returns the attacker score dict plus SAE diagnostics: reconstruction MSE,
    mean active-feature count (L0) at the identity-relevant coordinates, and
    the sparsity of features carrying identity-predictive weight (used for
    the mechanistic-explanation writeup, not the headline accuracy number).
    """
    sae, mean, std = sae_bundle
    train_codes = sae_encode(sae, x_train, mean, std, device=device)
    test_codes = sae_encode(sae, x_test, mean, std, device=device)

    probe = RidgeClassifier(alpha=10.0)
    probe.fit(train_codes, y_train)
    prediction = probe.predict(test_codes)
    score = _score(y_test, prediction)

    reconstruction, latent = sae(
        torch.tensor((x_defender_for_sae - mean) / std, dtype=torch.float32, device=device)
    )
    reconstruction_mse = float(
        torch.mean(
            (reconstruction - torch.tensor((x_defender_for_sae - mean) / std, dtype=torch.float32, device=device))
            ** 2
        )
    )
    mean_l0 = float((latent > 0).float().sum(dim=1).mean())

    # Concentration of identity-predictive weight across the sparse code:
    # sort features by |ridge coefficient| and ask how many are needed to
    # reach 50%/80%/95% of total predictive weight. This replaces an
    # arbitrary fixed threshold (which degenerately called ~all features
    # "active" when the code was not actually sparse) with a
    # threshold-free concentration curve, informative regardless of how
    # sparse the underlying code turns out to be.
    weight = np.abs(probe.coef_).sum(axis=0)
    order = np.argsort(-weight)
    sorted_weight = weight[order]
    total_weight = sorted_weight.sum()
    cumulative = np.cumsum(sorted_weight) / total_weight if total_weight > 0 else np.zeros_like(sorted_weight)
    latent_dim = weight.shape[0]

    def features_for(fraction):
        idx = int(np.searchsorted(cumulative, fraction))
        return min(idx + 1, latent_dim)

    top5pct_count = max(1, round(0.05 * latent_dim))
    top5pct_weight_fraction = float(cumulative[top5pct_count - 1]) if latent_dim > 0 else 0.0

    score["sae_diagnostics"] = {
        "latent_dim": int(latent_dim),
        "reconstruction_mse_on_defender_split": reconstruction_mse,
        "mean_active_features_per_example_l0": mean_l0,
        "mean_active_fraction_l0": mean_l0 / latent_dim if latent_dim else 0.0,
        "chosen_l1_weight": getattr(sae, "chosen_l1_weight", None),
        "target_l0_fraction": getattr(sae, "target_l0", None),
        "features_for_50pct_predictive_weight": features_for(0.5),
        "features_for_80pct_predictive_weight": features_for(0.8),
        "features_for_95pct_predictive_weight": features_for(0.95),
        "top_5pct_features_predictive_weight_fraction": top5pct_weight_fraction,
    }
    return score
