"""End-to-end pipeline: preprocess -> autoencoder -> K-Means -> AI analysis -> JSON for the dashboard.

Usage (from the project root):
    python ml/run_pipeline.py                         # original dataset
    python ml/run_pipeline.py --data path/to/file.csv # any dataset with the same columns
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from analysis import percentile_rank, response_propensity, similar_customers, soft_membership  # noqa: E402
from autoencoder import train_autoencoder  # noqa: E402
from clustering import (  # noqa: E402
    compare_methods, evaluate_k, fit_final_kmeans, profile_and_name, project_2d,
)
from preprocess import (  # noqa: E402
    CMP_COLS, MNT_COLS, OPTIONAL_COLUMNS, REQUIRED_COLUMNS, DatasetError,
    build_feature_matrix, clean_and_engineer, load_data,
)

ROOT = Path(__file__).resolve().parent.parent
ORIGINAL_DATA = ROOT / "data" / "marketing_campaign_modified.csv"
MODELS_DIR = ROOT / "models"
OUT_DIR = ROOT / "dashboard" / "data"

CUSTOMER_FIELDS = [
    "ID", "Age", "Age_Group", "Education", "Marital_Status", "Income", "Income_Band",
    "Total_Children", "Family_Size", "Dt_Customer", "Recency", *MNT_COLS, "Total_Spending",
    "NumDealsPurchases", "NumWebPurchases", "NumCatalogPurchases", "NumStorePurchases",
    "NumWebVisitsMonth", "Total_Purchases", *CMP_COLS, "Total_Campaigns_Accepted",
    "Complain", "Response", "Customer_Tenure_Days",
]
INT_FIELDS = ["ID", "Age", "Total_Children", "Family_Size", "Recency", *MNT_COLS, "Total_Spending",
              "NumDealsPurchases", "NumWebPurchases", "NumCatalogPurchases", "NumStorePurchases",
              "NumWebVisitsMonth", "Total_Purchases", *CMP_COLS, "Total_Campaigns_Accepted",
              "Complain", "Response", "Customer_Tenure_Days"]


def _relabel_by_spending(labels: np.ndarray, spending: np.ndarray) -> tuple[np.ndarray, list[int]]:
    """Make cluster ids stable and readable: 0 = highest average spending.

    Returns the new labels and `order`, where new cluster j was old cluster order[j].
    """
    means = {c: spending[labels == c].mean() for c in np.unique(labels)}
    order = sorted(means, key=means.get, reverse=True)
    mapping = {old: new for new, old in enumerate(order)}
    return np.vectorize(mapping.get)(labels), [int(o) for o in order]


def _write_atomic(outputs: dict[str, object]) -> None:
    """Write every output to a temp file first, then swap them all in at the end."""
    temps = []
    for name, payload in outputs.items():
        tmp = OUT_DIR / f".{name}.tmp"
        if isinstance(payload, Path):
            shutil.copyfile(payload, tmp)
        else:
            tmp.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
        temps.append((tmp, OUT_DIR / name))
    for tmp, final in temps:
        os.replace(tmp, final)


def run(data_path: Path, display_name: str | None = None) -> None:
    t0 = time.time()
    MODELS_DIR.mkdir(exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    display_name = display_name or data_path.name

    raw = load_data(data_path)
    n_raw = len(raw)
    raw_columns = [c for c in raw.columns if c != "_row"]
    df = clean_and_engineer(raw)
    X, feature_names = build_feature_matrix(df, MODELS_DIR / "scaler.pkl")

    ae = train_autoencoder(X, MODELS_DIR / "encoder.keras")
    latent = ae["latent"]

    k_metrics, best_k = evaluate_k(latent)
    km = fit_final_kmeans(latent, best_k, MODELS_DIR / "kmeans.pkl")
    labels, order = _relabel_by_spending(km.labels_, df["Total_Spending"].to_numpy())
    centers = km.cluster_centers_[order]
    comparison = compare_methods(X, latent, best_k)
    xy = project_2d(latent)
    profiles, names = profile_and_name(df, labels)

    # ---- AI analysis -----------------------------------------------------------------
    print("[analysis] computing memberships, neighbours and response propensity", flush=True)
    probs = soft_membership(latent, centers, labels)
    ids = df["ID"].to_numpy()
    similar = similar_customers(latent, ids)
    prop = response_propensity(X, df["Response"].to_numpy(), feature_names, MODELS_DIR / "response_model.pkl")
    recon_pct = percentile_rank(ae["recon_error"])

    # ---- customers.json -------------------------------------------------------------
    out = df[CUSTOMER_FIELDS].copy()
    out[INT_FIELDS] = out[INT_FIELDS].round(0).astype(int)
    out["Dt_Customer"] = df["Dt_Customer"].dt.strftime("%Y-%m-%d")
    out["Income"] = out["Income"].round(0).astype(int)
    out["row"] = df["_row"].astype(int)
    out["cluster_id"] = labels.astype(int)
    out["cluster_name"] = [names[int(c)] for c in labels]
    out["latent_x"] = np.round(xy[:, 0], 4)
    out["latent_y"] = np.round(xy[:, 1], 4)
    out["recon_error"] = np.round(ae["recon_error"], 5)
    out["recon_pct"] = np.round(recon_pct, 1)
    out["seg_probs"] = [[round(float(p), 3) for p in row] for row in probs]
    out["similar"] = similar
    if prop["available"]:
        out["resp_prob"] = np.round(prop["probs"], 4)
        out["drivers"] = prop["drivers"]
    records = out.to_dict(orient="records")

    # ---- model_metrics.json ---------------------------------------------------------
    chosen = next(r for r in k_metrics if r["k"] == best_k)
    metrics = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "dataset": {
            "name": display_name,
            "is_original": data_path.resolve() == ORIGINAL_DATA.resolve(),
            "rows": n_raw,
            "columns": raw_columns,
            "file": "data/dataset.csv",
        },
        "required_columns": REQUIRED_COLUMNS,
        "optional_columns": OPTIONAL_COLUMNS,
        "n_raw_rows": n_raw,
        "n_customers": len(df),
        "n_removed": n_raw - len(df),
        "n_features": len(feature_names),
        "feature_names": feature_names,
        "latent_dim": int(latent.shape[1]),
        "k_metrics": k_metrics,
        "chosen_k": best_k,
        "chosen_silhouette": chosen["silhouette"],
        "comparison": comparison,
        "loss_history": ae["history"],
        "cluster_profiles": profiles,
        "propensity": {k: v for k, v in prop.items() if k not in ("probs", "drivers")},
        "date_range": {"min": df["Dt_Customer"].min().strftime("%Y-%m-%d"),
                       "max": df["Dt_Customer"].max().strftime("%Y-%m-%d")},
    }

    print("[output] writing dashboard data", flush=True)
    _write_atomic({"customers.json": records, "model_metrics.json": metrics, "dataset.csv": data_path})

    # ---- console summary ------------------------------------------------------------
    line = "=" * 64
    print(f"\n{line}\n CUSTOMER SEGMENTATION PIPELINE - SUMMARY\n{line}")
    print(f" Dataset           : {display_name}")
    print(f" Customers         : {len(df):,} (removed {n_raw - len(df)} outliers)")
    print(f" Model features    : {len(feature_names)} -> latent {latent.shape[1]}-D")
    print(f" AE epochs         : {len(ae['history']['loss'])} (best {ae['history']['best_epoch']})")
    print(f" Chosen k          : {best_k}")
    print(f" Silhouette (AE+KM): {chosen['silhouette']:.4f}")
    print(f" Response rate     : {df['Response'].mean():.1%}")
    if prop["available"]:
        print(f" Propensity AUC    : {prop['auc']:.3f}")
    print(f" Date range        : {metrics['date_range']['min']} -> {metrics['date_range']['max']}")
    print(" Segments:")
    for p in profiles:
        print(f"   [{p['cluster_id']}] {p['name']:<26} n={p['size']:>4}  income={p['income']:>8,.0f}  "
              f"spend={p['spending']:>6,.0f}  response={p['response_rate']:.1%}")
    print(" Model comparison:")
    for row in comparison:
        print(f"   {row['method']:<34} sil={row['silhouette']:.3f}  DB={row['davies_bouldin']:.3f}  "
              f"CH={row['calinski_harabasz']:,.0f}")
    print(f" Output            : {OUT_DIR}")
    print(f" Done in {time.time() - t0:.1f}s\n{line}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=ORIGINAL_DATA, help="CSV/TSV dataset to analyse")
    parser.add_argument("--name", default=None, help="display name for the dataset")
    args = parser.parse_args()
    try:
        run(args.data, args.name)
    except DatasetError as exc:
        print(f"[error] {exc}", flush=True)
        sys.exit(2)


if __name__ == "__main__":
    main()
