"""K-Means selection, baseline comparison, 2-D projection and cluster profiling/naming."""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.mixture import GaussianMixture

SEED = 42
K_RANGE = range(2, 9)
K_CHOICE_RANGE = range(3, 7)  # keep the segment count useful for the business


def _scores(X: np.ndarray, labels: np.ndarray) -> dict:
    return {
        "silhouette": float(silhouette_score(X, labels)),
        "davies_bouldin": float(davies_bouldin_score(X, labels)),
        "calinski_harabasz": float(calinski_harabasz_score(X, labels)),
    }


def evaluate_k(latent: np.ndarray) -> tuple[list[dict], int]:
    """Fit K-Means for each k, record metrics and pick k by best silhouette within 3..6."""
    rows = []
    for k in K_RANGE:
        km = KMeans(n_clusters=k, n_init=20, random_state=SEED).fit(latent)
        rows.append({"k": k, "inertia": float(km.inertia_), **_scores(latent, km.labels_)})
        print(f"[kmeans] k={k}  inertia={km.inertia_:10.1f}  silhouette={rows[-1]['silhouette']:.4f}")
    candidates = [r for r in rows if r["k"] in K_CHOICE_RANGE]
    best_k = max(candidates, key=lambda r: r["silhouette"])["k"]
    print(f"[kmeans] chosen k = {best_k} (best silhouette in {K_CHOICE_RANGE.start}..{K_CHOICE_RANGE.stop - 1})")
    return rows, best_k


def fit_final_kmeans(latent: np.ndarray, k: int, model_path: str | Path) -> KMeans:
    km = KMeans(n_clusters=k, n_init=20, random_state=SEED).fit(latent)
    joblib.dump(km, model_path)
    return km


def compare_methods(X_scaled: np.ndarray, latent: np.ndarray, k: int) -> list[dict]:
    """Score five clustering approaches at the chosen k, each in the space it clusters."""
    pca_space = PCA(n_components=min(8, X_scaled.shape[1]), random_state=SEED).fit_transform(X_scaled)
    methods = [
        ("K-Means (raw scaled features)", "Raw scaled", X_scaled,
         KMeans(n_clusters=k, n_init=20, random_state=SEED).fit_predict(X_scaled)),
        ("Autoencoder + K-Means (proposed)", "AE latent (8-D)", latent,
         KMeans(n_clusters=k, n_init=20, random_state=SEED).fit_predict(latent)),
        ("PCA (8) + K-Means", "PCA (8-D)", pca_space,
         KMeans(n_clusters=k, n_init=20, random_state=SEED).fit_predict(pca_space)),
        ("Gaussian Mixture (latent)", "AE latent (8-D)", latent,
         GaussianMixture(n_components=k, random_state=SEED, n_init=3).fit_predict(latent)),
        ("Agglomerative Ward (latent)", "AE latent (8-D)", latent,
         AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(latent)),
    ]
    table = []
    for name, space, X, labels in methods:
        if len(np.unique(labels)) < 2:
            continue
        table.append({"method": name, "space": space, **_scores(X, labels),
                      "proposed": "proposed" in name})
    return table


def project_2d(latent: np.ndarray) -> np.ndarray:
    return PCA(n_components=2, random_state=SEED).fit_transform(latent)


PROFILE_COLS = {
    "income": "Income", "spending": "Total_Spending", "children": "Total_Children",
    "age": "Age", "deals": "NumDealsPurchases", "web_purchases": "NumWebPurchases",
    "web_share": "Web_Share", "campaigns": "Total_Campaigns_Accepted", "response_rate": "Response",
}


# Each persona has a qualifying rule and a fit score, both based on ratios of the cluster
# mean to the overall mean (1.0 = average). "Mainstream Customers" is the catch-all.
PERSONAS = {
    "Premium Shoppers": (lambda r: r["income"] >= 1.1 and r["spending"] >= 1.3,
                         lambda r: (r["income"] - 1) + (r["spending"] - 1)),
    "Campaign Enthusiasts": (lambda r: r["campaigns"] >= 1.6,
                             lambda r: r["campaigns"] - 1),
    "Budget Families": (lambda r: r["spending"] <= 0.7 and r["children"] >= 1.0,
                        lambda r: (1 - r["spending"]) + (r["children"] - 1) + max(r["deals"] - 1, 0)),
    "Digital-First Buyers": (lambda r: r["web_share"] >= 1.1,
                             lambda r: 3 * (r["web_share"] - 1)),
    "Mainstream Customers": (lambda r: True,
                             lambda r: -(abs(r["income"] - 1) + abs(r["spending"] - 1))),
}
SUFFIXES = ["", " II", " III", " IV", " V", " VI"]


def _best_persona(r: dict) -> str:
    qualifying = [p for p, (ok, _) in PERSONAS.items() if ok(r)]
    return max(qualifying, key=lambda p: PERSONAS[p][1](r))


def _assign_names(ratios: dict[int, dict]) -> dict[int, str]:
    """Greedy unique assignment: the strongest qualifying (cluster, persona) match is fixed first.

    Clusters left without an unused qualifying persona take their best-fitting persona
    with a roman-numeral suffix ("II", "III", ...) so names stay unique.
    """
    pairs = sorted(
        ((score(r), cid, persona) for cid, r in ratios.items()
         for persona, (ok, score) in PERSONAS.items() if ok(r)),
        reverse=True,
    )
    names: dict[int, str] = {}
    used: set[str] = set()
    for _, cid, persona in pairs:
        if cid not in names and persona not in used:
            names[cid] = persona
            used.add(persona)
    counts = {n: 1 for n in names.values()}
    for cid, r in ratios.items():
        if cid in names:
            continue
        best = _best_persona(r)
        counts[best] = counts.get(best, 0) + 1
        names[cid] = best + SUFFIXES[min(counts[best] - 1, len(SUFFIXES) - 1)]
    return names


def profile_and_name(df: pd.DataFrame, labels: np.ndarray) -> tuple[list[dict], dict[int, str]]:
    """Compute per-cluster means and give each cluster a unique business name."""
    df = df.assign(cluster_id=labels)
    overall = {key: float(df[col].mean()) for key, col in PROFILE_COLS.items()}
    means_by, ratios_by, sizes = {}, {}, {}
    for cid, sub in df.groupby("cluster_id"):
        means = {key: float(sub[col].mean()) for key, col in PROFILE_COLS.items()}
        means_by[int(cid)] = means
        ratios_by[int(cid)] = {k: (means[k] / overall[k] if overall[k] else 1.0) for k in means}
        sizes[int(cid)] = int(len(sub))
    names = _assign_names(ratios_by)
    profiles = [
        {"cluster_id": cid, "name": names[cid], "size": sizes[cid], "share": sizes[cid] / len(df),
         **means_by[cid], "ratios": ratios_by[cid]}
        for cid in sorted(means_by)
    ]
    return profiles, names
