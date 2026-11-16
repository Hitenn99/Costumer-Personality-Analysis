"""Per-customer AI analysis used by the dashboard's "AI Analysis" page.

* Soft segment membership from latent distances to each K-Means centroid.
* Most similar customers (nearest neighbours in the 8-D latent space).
* Response propensity from a logistic-regression model (out-of-fold probabilities,
  so each customer's score comes from a model that never saw that customer).
* Per-customer drivers of the propensity score (coefficient x scaled feature value).
* Reconstruction-error percentile (how unusual the customer is for the autoencoder).
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.neighbors import NearestNeighbors

SEED = 42
N_SIMILAR = 5
N_DRIVERS = 4


def soft_membership(latent: np.ndarray, centers: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Gaussian soft assignment: p_k ~ exp(-d_k^2 / 2 sigma^2), sigma^2 = mean within-cluster d^2."""
    d2 = ((latent[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
    sigma2 = float(d2[np.arange(len(latent)), labels].mean()) or 1.0
    logits = -d2 / (2 * sigma2)
    logits -= logits.max(axis=1, keepdims=True)
    probs = np.exp(logits)
    return probs / probs.sum(axis=1, keepdims=True)


def similar_customers(latent: np.ndarray, ids: np.ndarray) -> list[list]:
    """For each customer, the N_SIMILAR nearest other customers as [id, match score 0..1]."""
    nn = NearestNeighbors(n_neighbors=min(N_SIMILAR + 1, len(latent))).fit(latent)
    dist, idx = nn.kneighbors(latent)
    scale = float(np.median(dist[:, 1:])) or 1.0  # typical neighbour distance
    out = []
    for i in range(len(latent)):
        pairs = [(j, d) for j, d in zip(idx[i], dist[i]) if j != i][:N_SIMILAR]
        out.append([[int(ids[j]), round(float(np.exp(-d / (2 * scale))), 3)] for j, d in pairs])
    return out


def response_propensity(X: np.ndarray, y: np.ndarray, feature_names: list[str],
                        model_path: str | Path) -> dict:
    """Fit a response model; return out-of-fold probabilities, CV AUC and per-customer drivers."""
    n_pos = int(y.sum())
    if n_pos < 5 or n_pos > len(y) - 5:
        print("[propensity] skipped: Response needs at least 5 responders and 5 non-responders", flush=True)
        return {"available": False}

    model = LogisticRegression(C=0.5, max_iter=2000)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    oof = cross_val_predict(model, X, y, cv=cv, method="predict_proba")[:, 1]
    auc = float(roc_auc_score(y, oof))
    model.fit(X, y)
    joblib.dump(model, model_path)

    contrib = X * model.coef_[0]  # log-odds contribution of each scaled feature
    top = np.argsort(-np.abs(contrib), axis=1)[:, :N_DRIVERS]
    drivers = [[[feature_names[j], round(float(contrib[i, j]), 3)] for j in top[i]] for i in range(len(X))]
    print(f"[propensity] logistic regression, 5-fold CV AUC = {auc:.3f}", flush=True)
    return {
        "available": True,
        "model": "Logistic regression (5-fold out-of-fold probabilities)",
        "auc": auc,
        "positive_rate": float(y.mean()),
        "probs": oof,
        "drivers": drivers,
    }


def percentile_rank(values: np.ndarray) -> np.ndarray:
    """0..100 percentile of each value within the array."""
    ranks = values.argsort().argsort()
    return ranks / max(len(values) - 1, 1) * 100
