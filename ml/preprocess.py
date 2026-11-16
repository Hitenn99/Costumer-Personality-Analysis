"""Data loading, cleaning and feature engineering for the customer segmentation pipeline."""
from __future__ import annotations

import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

EXPECTED_COLUMNS = [
    "ID", "Year_Birth", "Education", "Marital_Status", "Income", "Kidhome", "Teenhome",
    "Dt_Customer", "Recency", "MntWines", "MntFruits", "MntMeatProducts", "MntFishProducts",
    "MntSweetProducts", "MntGoldProds", "NumDealsPurchases", "NumWebPurchases",
    "NumCatalogPurchases", "NumStorePurchases", "NumWebVisitsMonth", "AcceptedCmp1",
    "AcceptedCmp2", "AcceptedCmp3", "AcceptedCmp4", "AcceptedCmp5", "Complain",
    "Z_CostContact", "Z_Revenue", "Response",
]
MNT_COLS = ["MntWines", "MntFruits", "MntMeatProducts", "MntFishProducts", "MntSweetProducts", "MntGoldProds"]
PURCHASE_COLS = ["NumWebPurchases", "NumCatalogPurchases", "NumStorePurchases"]
CMP_COLS = ["AcceptedCmp1", "AcceptedCmp2", "AcceptedCmp3", "AcceptedCmp4", "AcceptedCmp5"]

AGE_BINS = [-np.inf, 30, 45, 60, 75, np.inf]
AGE_LABELS = ["<30", "30-44", "45-59", "60-74", "75+"]
INCOME_BINS = [-np.inf, 30_000, 60_000, 90_000, np.inf]
INCOME_LABELS = ["Low (<30k)", "Mid (30-60k)", "High (60-90k)", "Very High (90k+)"]

EDUCATION_MAP = {
    "Basic": "Undergraduate", "2n Cycle": "Undergraduate",
    "Graduation": "Graduate", "Master": "Postgraduate", "PhD": "PhD",
}
MARITAL_MAP = {
    "Married": "Couple", "Together": "Couple",
    "Single": "Single", "Divorced": "Single", "Widow": "Single",
    "Alone": "Single", "Absurd": "Single", "YOLO": "Single",
}

# Numeric columns fed to the autoencoder (Response is deliberately excluded).
NUMERIC_FEATURES = [
    "Age", "Income", "Kidhome", "Teenhome", "Recency", *MNT_COLS,
    "NumDealsPurchases", *PURCHASE_COLS, "NumWebVisitsMonth", *CMP_COLS, "Complain",
    "Total_Spending", "Total_Purchases", "Total_Children", "Has_Partner", "Family_Size",
    "Customer_Tenure_Days", "Total_Campaigns_Accepted",
]
# Heavily right-skewed money columns get a log1p before scaling so a few big spenders
# do not dominate the distance metric used by the autoencoder and K-Means.
LOG_FEATURES = ["Income", *MNT_COLS, "Total_Spending"]


# Columns that can be safely defaulted when absent; every other expected column is required.
OPTIONAL_COLUMNS = ["ID", *CMP_COLS, "Complain", "Z_CostContact", "Z_Revenue"]
REQUIRED_COLUMNS = [c for c in EXPECTED_COLUMNS if c not in OPTIONAL_COLUMNS]
MIN_ROWS = 50


class DatasetError(ValueError):
    """Raised when an input file cannot be used for the pipeline."""


def read_table(path: str | Path, **kwargs) -> pd.DataFrame:
    """Read a CSV/TSV with automatic delimiter detection and stripped column names."""
    df = pd.read_csv(path, sep=None, engine="python", **kwargs)
    df.columns = df.columns.astype(str).str.strip()
    return df


def validate_columns(columns) -> list[str]:
    """Return the required columns that are missing."""
    present = {str(c).strip() for c in columns}
    return [c for c in REQUIRED_COLUMNS if c not in present]


def load_data(path: str | Path) -> pd.DataFrame:
    """Read the dataset, fail clearly on missing required columns and default optional ones."""
    try:
        df = read_table(path)
    except Exception as exc:  # malformed file
        raise DatasetError(f"Could not read {Path(path).name} as a CSV/TSV file: {exc}") from exc
    missing = validate_columns(df.columns)
    if missing:
        raise DatasetError(f"Missing required columns: {', '.join(missing)}")

    absent = [c for c in OPTIONAL_COLUMNS if c not in df.columns]
    if absent:
        warnings.warn(f"Optional columns missing (defaulted): {absent}")
        for col in absent:
            df[col] = np.arange(1, len(df) + 1) if col == "ID" else 0
    if df["ID"].duplicated().any():
        warnings.warn("Duplicate IDs found; customers are renumbered 1..n so every ID is unique.")
        df["ID"] = np.arange(1, len(df) + 1)

    # Keep the raw row position so the dashboard can join raw rows to their segment.
    df["_row"] = np.arange(len(df))
    print(f"[load] {len(df):,} rows x {df.shape[1] - 1} columns from {Path(path).name}", flush=True)
    return df


def _parse_dates(series: pd.Series) -> pd.Series:
    """Parse Dt_Customer trying day-first first, then a set of common explicit formats."""
    raw = series.astype(str).str.strip()
    parsed = pd.to_datetime(raw, dayfirst=True, errors="coerce", format="mixed")
    for fmt in ("%d-%m-%Y", "%d/%m/%Y", "%Y-%m-%d", "%m/%d/%Y", "%d.%m.%Y"):
        if not parsed.isna().any():
            break
        parsed = parsed.fillna(pd.to_datetime(raw, format=fmt, errors="coerce"))
    if parsed.isna().any():
        warnings.warn(f"{int(parsed.isna().sum())} Dt_Customer values could not be parsed; using the median date.")
        parsed = parsed.fillna(parsed.median())
    return parsed


def clean_and_engineer(df: pd.DataFrame) -> pd.DataFrame:
    """Impute, remove outliers, engineer features and simplify categories."""
    df = df.copy()
    # Drop pre-computed helper columns from the modified file; Age is recomputed below.
    df = df.drop(columns=[c for c in ("Age", "AgeCategory") if c in df.columns])

    df["Income"] = pd.to_numeric(df["Income"], errors="coerce")
    n_missing_income = int(df["Income"].isna().sum())
    df["Income"] = df["Income"].fillna(df["Income"].median())
    print(f"[clean] filled {n_missing_income} missing Income values with the median", flush=True)

    # Uploaded files may contain blanks or text in count columns: coerce and default to 0.
    count_cols = [c for c in EXPECTED_COLUMNS
                  if c not in ("ID", "Education", "Marital_Status", "Dt_Customer", "Income")]
    for col in count_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    df["Response"] = (df["Response"] > 0).astype(int)
    df["Year_Birth"] = df["Year_Birth"].replace(0, np.nan).fillna(df["Year_Birth"].median()).astype(int)
    df["Education"] = df["Education"].astype(str).str.strip()
    df["Marital_Status"] = df["Marital_Status"].astype(str).str.strip()

    df["Dt_Customer"] = _parse_dates(df["Dt_Customer"])
    latest = df["Dt_Customer"].max()
    df["Age"] = latest.year - df["Year_Birth"]

    n_before = len(df)
    age_out = df["Age"] > 100
    inc_out = df["Income"] > 600_000
    df = df[~(age_out | inc_out)].reset_index(drop=True)
    print(f"[clean] removed {n_before - len(df)} outlier rows "
          f"(Age>100: {int(age_out.sum())}, Income>600k: {int(inc_out.sum())}) -> {len(df):,} rows", flush=True)
    if len(df) < MIN_ROWS:
        raise DatasetError(f"Only {len(df)} usable rows after cleaning; at least {MIN_ROWS} are needed.")

    df["Total_Spending"] = df[MNT_COLS].sum(axis=1)
    df["Total_Purchases"] = df[PURCHASE_COLS].sum(axis=1)
    df["Total_Children"] = df["Kidhome"] + df["Teenhome"]
    df["Has_Partner"] = df["Marital_Status"].isin(["Married", "Together"]).astype(int)
    df["Family_Size"] = 1 + df["Has_Partner"] + df["Total_Children"]
    df["Customer_Tenure_Days"] = (latest - df["Dt_Customer"]).dt.days
    df["Total_Campaigns_Accepted"] = df[CMP_COLS].sum(axis=1)
    df["Web_Share"] = np.where(df["Total_Purchases"] > 0,
                               df["NumWebPurchases"] / df["Total_Purchases"].replace(0, 1), 0.0)
    df["Age_Group"] = pd.cut(df["Age"], AGE_BINS, labels=AGE_LABELS, right=False).astype(str)
    df["Income_Band"] = pd.cut(df["Income"], INCOME_BINS, labels=INCOME_LABELS, right=False).astype(str)

    df["Education"] = df["Education"].map(EDUCATION_MAP).fillna("Other")
    df["Marital_Status"] = df["Marital_Status"].map(MARITAL_MAP).fillna("Single")

    df = df.drop(columns=[c for c in ("Z_CostContact", "Z_Revenue") if c in df.columns])
    return df


def build_feature_matrix(df: pd.DataFrame, scaler_path: str | Path) -> tuple[np.ndarray, list[str]]:
    """Assemble model features (numeric + one-hot categoricals), scale them and save the scaler."""
    num = df[NUMERIC_FEATURES].astype(float).copy()
    for col in LOG_FEATURES:
        num[col] = np.log1p(num[col].clip(lower=0))
    cats = pd.get_dummies(df[["Education", "Marital_Status"]], prefix=["Edu", "Marital"], dtype=float)
    features = pd.concat([num, cats], axis=1)

    scaler = StandardScaler()
    X = scaler.fit_transform(features.values)
    joblib.dump(scaler, scaler_path)
    print(f"[features] {X.shape[1]} model features; scaler saved to {Path(scaler_path).name}")
    return X.astype(np.float32), list(features.columns)
