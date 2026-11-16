# Customer Personality Analysis Dashboard

This project segments the customers in `marketing_campaign_modified.csv` (2,240 rows). An **autoencoder** compresses each customer into an 8-D latent representation, and **K-Means** clusters that latent space. An interactive, cross-filtering dashboard (plain HTML/CSS/JS with Apache ECharts) shows the result. Campaign **Response** plays the role that "attrition" plays in a classic HR dashboard.

```
customer-dashboard/
├── data/marketing_campaign_modified.csv   input
├── ml/
│   ├── preprocess.py      load, clean, feature engineering, scaling
│   ├── autoencoder.py     Keras autoencoder (32-16-8 encoder, mirrored decoder)
│   ├── clustering.py      k selection, baseline comparison, profiling, auto-naming
│   ├── analysis.py        per-customer AI analysis (membership, propensity, neighbours)
│   └── run_pipeline.py    runs everything and writes dashboard/data/*
├── models/                scaler.pkl, encoder.keras, kmeans.pkl, response_model.pkl
├── dashboard/             index.html, styles.css, app.js (dashboard + router),
│                          dataset.js, ai.js, data/ (customers.json, model_metrics.json, dataset.csv)
├── server.py              app server: static files + upload/retrain API
└── requirements.txt
```

## Setup

TensorFlow 2.17 needs **Python 3.10–3.12**. Python 3.13+ is not supported.

```powershell
cd customer-dashboard
py -3.11 -m venv .venv            # macOS/Linux: python3.11 -m venv .venv
.\.venv\Scripts\activate          # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

## Run the ML pipeline

```powershell
python ml/run_pipeline.py
```

This takes about 25 seconds on a CPU. It writes `dashboard/data/customers.json` and `dashboard/data/model_metrics.json`, saves the models to `models/`, and prints a summary.

## Start the app

```powershell
python server.py
```

Then open <http://localhost:8000>. `server.py` uses only the Python standard library. It serves the dashboard and adds the upload API that the **Dataset** page uses. `python -m http.server` (run from `dashboard/`) still shows all three pages read-only, but uploading needs `server.py`. Opening `index.html` directly from disk does not work, because browsers block `fetch()` on `file://` URLs.

## Pages

| Page | What it does |
|---|---|
| **Dashboard** (`#/dashboard`) | The cross-filtering segmentation dashboard. On screens of about 1500×860 or larger it fits one screen without scrolling. |
| **Dataset** (`#/dataset`) | The complete uploaded dataset: summary tiles, plus a table you can search, sort and page through. Each row is tagged with its segment, or marked *Excluded* if it was an outlier, and missing values are highlighted. Click a row to analyse that customer. **Upload file** retrains everything on a new CSV/TSV, **Restore original dataset** switches back, and **Download CSV** exports the current file. |
| **AI Analysis** (`#/ai?id=5524`) | Pick a customer to see their predicted segment, confidence spread across all segments, and a written summary. It also shows their chance of responding to a campaign and what drives it, comparisons with their segment and with all customers, their position in the latent space, the most similar customers, an anomaly check, and recommended actions. |

### Uploading a new dataset

1. Open **Dataset**, click **Upload file** and choose a `.csv`, `.tsv` or `.txt` file. The delimiter is detected automatically.
2. The server checks the file before it starts training. It needs the required columns listed at the bottom of the Dataset page and at least 50 rows. If anything is missing, the file is rejected with a message that lists the missing columns.
3. The pipeline retrains the autoencoder, K-Means and the response model in a background process, which takes about 30 seconds. A progress bar shows each stage.
4. When it finishes, the app reloads and all three pages show the new data. If training fails, the previous data stays in place, because outputs are only swapped in after a successful run.

Uploads are kept in `data/uploads/`. The file in use is copied to `dashboard/data/dataset.csv`. Optional columns (`ID`, `AcceptedCmp1-5`, `Complain`, `Z_*`) are filled with defaults when they are missing.

### How the AI analysis works

Everything is computed by `ml/analysis.py` while the pipeline runs, so the page stays fast:

- **Category and confidence:** the customer's 8-D autoencoder embedding is compared with each K-Means centroid. Confidence is a Gaussian soft assignment based on those distances.
- **Response propensity:** a logistic-regression model predicts `Response` (5-fold CV AUC ≈ 0.90 on the original data). Each customer's score comes from the fold that did *not* train on them. "Drivers" are that customer's largest coefficient × feature contributions.
- **Similar customers:** the 5 nearest neighbours in the latent space.
- **Anomaly check:** the customer's reconstruction-error percentile.
- **Summary and actions:** generated from these numbers, plus the averages for the segment and for all customers.

## Method

**Preprocessing**
1. Missing `Income` values (23) are filled with the median.
2. Age is calculated as the latest enrolment year minus `Year_Birth`. Rows with Age > 100 (3 rows) or Income > 600k (1 row) are removed, leaving 2,236 customers.
3. Engineered features: total spending, total purchases, children, partner and family size, tenure, campaigns accepted, and web share.
4. Education and marital status are simplified, then one-hot encoded.
5. Money columns get a `log1p` before `StandardScaler`, so a few very large spenders do not dominate the distances.
6. `Response` is never used as an input. It is kept aside so the segments can be checked against it.

**Why an autoencoder?** PCA can only find linear combinations of features. An autoencoder learns a non-linear compression: for example, the interactions between income, family size, channel mix and spending. The bottleneck also has to discard noise, so it keeps only the structure it needs to rebuild the input. The 35 scaled features pass through `32 → BatchNorm → 16 → 8 (latent)` and a mirrored decoder, trained with MSE, Adam (lr 1e-3) and early stopping. Each customer's reconstruction error is reused in the **Anomaly Watch** card: customers the model rebuilds badly are unusual, for example a very high income with almost no spending.

**Choosing k.** K-Means is run on the latent space for k = 2..8, recording inertia (elbow), Silhouette, Davies-Bouldin and Calinski-Harabasz each time. k is the best Silhouette within **3–6**, a range that keeps the segments useful to the business. That range excludes k = 2, which scores higher but only splits customers into "spenders" and "non-spenders". The current run chooses **k = 4 (silhouette 0.344)**.

**Model comparison at k = 4.** Each method is scored in the space it clusters.

| Method | Silhouette ↑ | Davies-Bouldin ↓ | Calinski-Harabasz ↑ |
|---|---|---|---|
| K-Means on raw scaled features | 0.155 | 2.103 | 351 |
| **Autoencoder + K-Means** | **0.344** | **1.108** | **1,131** |
| PCA (8) + K-Means | 0.199 | 1.748 | 626 |
| GMM on latent | 0.070 | 2.658 | 288 |
| Agglomerative (Ward) on latent | 0.332 | 1.119 | 1,096 |

**Segment naming.** Each cluster's means are compared with the overall average as ratios. Each persona has a qualifying rule and a fit score: Premium = high income and spend, Budget Families = low spend with children, Digital-First = high web share, Campaign Enthusiasts = high campaign acceptance, and Mainstream is the catch-all. Names are assigned greedily from the strongest match down, so they stay unique. If there are more clusters than personas, a suffix ("II", "III") is added.

## The segments (current run)

| Segment | Customers | Avg income | Avg spend | Response rate | What it means |
|---|---|---|---|---|---|
| **Campaign Enthusiasts** | 121 | $79.9K | $1,558 | 66.9% | Affluent, few children, and they accept many campaigns. They are the most responsive audience by far. |
| **Premium Shoppers** | 284 | $65.8K | $1,100 | 29.9% | High spend across catalog, store and web. They respond well and are worth premium offers. |
| **Mainstream Customers** | 961 | $61.4K | $819 | 9.2% | Solid spenders who have never accepted a previous campaign. A good target for testing new messaging. |
| **Budget Families** | 870 | $33.2K | $77 | 9.2% | Lower income, more children at home, very low spend. Better reached with deals than with premium campaigns. |

The names and numbers are recomputed on every pipeline run, and the dashboard reads them from the JSON. Nothing in the front end is hard-coded.

## Dashboard features

- **Cross-filtering.** Use the filter chips or click a chart element: a donut, an age or education bar, or a latent-space point. Selections within one group combine with OR; separate groups combine with AND. Every data card recomputes in the browser.
- **Charts that ignore their own filter.** The age, education, marital and segment charts ignore the filter on their own dimension, so the other categories stay visible. Unselected categories are dimmed.
- **Static model charts.** The training loss, optimal-k and model comparison charts do not change with filters.
- **Empty state.** If no customers match the filters, a banner says so and offers a reset.
