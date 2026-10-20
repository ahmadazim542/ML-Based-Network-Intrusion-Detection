# ML-Based Network Intrusion Detection — a controlled benchmark

Five classifiers (Logistic Regression, Random Forest, XGBoost, RBF-SVM, MLP) compared on the
**corrected CICIDS2017** dataset under one identical, reproducible pipeline, with per-class
metrics, imbalance handling, significance tests, SHAP explainability, deployability
measurements, a cross-dataset test and a leave-one-attack-out test.

Proposal and slides are in [`docs/`](docs/).

## Setup

```bash
brew install python@3.12 libomp          # libomp is required by XGBoost on macOS
python3.12 -m venv .venv
.venv/bin/pip install -e ".[notebooks,onnx,dev]"
source .venv/bin/activate
```

Check everything works on synthetic data (about 2 minutes, writes to `results/smoke/`):

```bash
python -m ids.synthetic
ids all --config configs/smoke.yaml
pytest
```

## Getting the data

| Dataset | Role | Download | On disk |
|---|---|---|---|
| Corrected CICIDS2017 | train + test | 328 MB | ~1.15 GB |
| Corrected CSE-CIC-IDS2018 | cross-dataset test only | streams ~1 GB per day | a few hundred MB as parquet |

Both come from <https://intrusion-detection.distrinet-research.be/CNS2022/>. They were
re-extracted with a fixed CICFlowMeter and relabelled (Engelen et al. 2021; Liu et al. 2022),
which fixes known flow-construction and labelling errors in the original CSVs.

```bash
python scripts/fetch_data.py list          # show files and sizes in both archives
python scripts/fetch_data.py cicids2017    # -> data/raw/cicids2017_fixed/*.csv
caffeinate -i python scripts/fetch_data.py cicids2018   # 5 days, all attacks + 5% benign; ~1-2 h, keep the Mac awake
```

The full 2018 archive is 9.7 GB, so the fetch script never downloads it. It reads single day
files out of the remote zip with HTTP range requests and keeps only what's needed.

If you already have the CSVs, put them in `data/raw/cicids2017_fixed/` instead. The files are
`monday.csv` … `friday.csv`.

## Running

Each step caches its output, so steps can be re-run on their own:

```bash
ids prepare      # clean, split, select features (cached in data/processed/)
ids benchmark    # tune + 5-fold CV + final fit for 5 models x 3 imbalance strategies
ids stats        # corrected resampled t-test + McNemar between models
ids deploy       # size, single-flow latency, throughput, ONNX latency
ids explain      # SHAP global + local plots
ids cross        # evaluate on CSE-CIC-IDS2018 without retraining
ids loao         # leave-one-attack-out
ids report       # summary tables + figures
ids all          # everything above in order
```

Useful options are `--models xgboost random_forest`, `--strategies none smote`,
`--split time_block`, `--task binary` and `--skip-cv`. All settings live in
[`configs/default.yaml`](configs/default.yaml).

The notebooks in `notebooks/01…08` follow the proposal's phases. They call the same code and
display the results, so the analysis and the write-up stay in sync with the pipeline.

Outputs go to `results/<dataset>__<task>__<split>/`. This includes `summary.md`,
`significance.csv`, `deployability.csv`, `cross_dataset__*.csv`,
`leave_one_attack_out.csv`, the figures, `shap/`, per-run metrics/predictions in `runs/` and
the saved models in `models/`.

## Methodology decisions (and where they differ from the proposal)

| Decision | Why |
|---|---|
| Corrected CICIDS2017 instead of the original CSVs | The original has documented labelling and flow-construction errors. "Attempted" flows (attack sessions with no attack payload) count as benign, as the dataset authors recommend. |
| Cross-dataset target is corrected CSE-CIC-IDS2018, not NSL-KDD | NSL-KDD has 41 features from a different extractor, so its features cannot be aligned. CSE-CIC-IDS2018 uses the same fixed CICFlowMeter, so the features match exactly. The test is binary only, because the attack taxonomies differ. |
| IPs, ports, flow ID and timestamp are dropped | They identify testbed hosts, not attack behaviour. A model using them would not transfer. |
| Duplicates are removed **before** splitting; conflicting-label duplicates are dropped entirely | Otherwise copies land in both train and test and inflate scores. |
| Benign rows are capped at 600k | 8 GB RAM. All attack rows are kept. |
| Families under 100 flows (Heartbleed) are excluded from the multi-class task | With ~2 test flows, their per-class metrics would be noise. They are still used in the binary and leave-one-attack-out experiments, and the exclusion is reported. |
| Feature selection (constant, then \|r\| > 0.95, then RFE-RF) is fitted on the training split only, once | Every model sees the same features, and the test data is never seen. |
| SMOTE runs inside each CV fold (imblearn Pipeline) and only lifts minority classes to 20k rows | No leakage, and no flood of millions of synthetic rows. Neighbours are found in log-scaled space because raw flow features span ~10 orders of magnitude. |
| Signed-log + standardisation for LR/SVM/MLP; trees use raw features | Flow features are extremely heavy-tailed. Trees don't care, and keeping raw values keeps SHAP plots readable. |
| RBF-SVM trains on 60k rows; MLP on 400k | RBF-SVM cost grows roughly with n². The subsample is proportional with a per-class floor, so the imbalance is preserved. |
| Hyper-parameters are tuned once per model (RandomizedSearchCV, 12 draws, 150k rows) and reused for all imbalance strategies | Fixed, documented budget. The strategy comparison isn't confounded by different tuning. |
| Significance uses the Nadeau–Bengio corrected t-test plus McNemar, not a plain paired t-test | CV folds overlap, so a plain t-test overstates significance. |
| Also reported: PR-AUC and detection rate at FPR = 0.1% / 1% | These are more informative than ROC-AUC/accuracy under heavy imbalance, and closer to how an IDS is actually tuned. |
| New: leave-one-attack-out experiment | It directly tests the abstract's claim that ML detects attacks it hasn't seen. |
| New: `--split time_block` | Flows are grouped into 5-minute blocks so near-identical neighbouring flows can't straddle the split. Comparing against the random split measures how much the random split leaks. |
| Latency is measured single-threaded, plus ONNX runtime for tree models | This replaces the Raspberry Pi stretch goal. Note that it is classification cost only. Real detection delay is dominated by waiting for the flow to end (FIN/RST or a 120 s timeout). |

## Layout

```
configs/      default.yaml (real runs), smoke.yaml (synthetic)
src/ids/      datasets, preprocess, split, features, sampling, models, evaluate,
              stats, deploy, explain, plots, experiments, cli
scripts/      fetch_data.py
notebooks/    01_eda ... 08_generalization
tests/        end-to-end test on synthetic data
docs/         proposal and presentation
```
