# Campsite Price Imitator & Initializer

Machine-learning models that recommend the **opening price** for a campsite accommodation
week, learned from two years of anonymised booking data. Two questions are tackled:

- **Imitator** — reproduce the price the campsite historically *opened* a week with
  (`InitialPrice`).
- **Initializer** — recommend a price tied to a **demand target** rather than history:
  either the price at an *X%-occupancy* threshold (`TargetPrice`) or the price in market at
  the season's *demand lift-off lead time* (`LeadTimePrice`).

All reusable logic lives in `src/`; the notebooks import, call and display.

> **Scope:** the model notebook is currently a **low-season (LS) spring proof-of-concept**.
> A *real* prior-year price exists only for 2025→2026 arrivals and the 2026 extract only
> covers spring, so the held-out test set is LS-only. Metrics should be read as a PoC, not a
> general all-season model.

---

## Data

`data/raw/Mekong_Delta_Multiple_campsites_datasets_capacity_over_200_anonymised_final.csv`
— an **anonymised** weekly booking extract (~1.2M snapshot rows, 3,842 reservable options,
37 campsites, arrival years 2024–2026, 5 seasonal clusters HS/LS/S1/S2/WTR). For every
option it records the discounted price and cumulative bookings at each *weeks-before-arrival*
horizon. The raw file is never modified.

---

## Notebooks

### `notebooks/Price_imitator_EDA.ipynb` — Exploratory Data Analysis
Two parts:
- **Part 1 — The Data Story** (visual): price distribution & skew, the booking curve, price
  decline from the opening price (with cuts/raises), the accommodation-range ladder,
  last-year vs this-year price, campsite price ranking, **inter-campsite variability for the
  same product**, demand-vs-price and sell-out timing, and a correlation heatmap.
- **Part 2 — Methodology & Data Quality**: grain validation, market-group structure,
  monotonicity/overbooking/missingness, prior-year completeness, feature diagnostics,
  seasonal structure and the target definitions — each with a `➡ Decision` note linking the
  finding to a modelling choice.

### `notebooks/Price_imitator.ipynb` — Modelling pipeline
Builds the option-week feature table and runs Ridge / Lasso / Random-Forest models against
global-mean and **segment-mean** baselines, with `TimeSeriesSplit` CV and a log-price target.
Steps:
1. Setup & load → 2. Derive `InitialPrice` → 3. Train the **imitator** (`InitialPrice`) →
4. Coverage analysis → 5–6. Derive `TargetPrice` → 7. Train the **occupancy initializer** →
8. **Lead-time initializer** (`LeadTimePrice`) → 9. **Corrective** on-target experiment →
10. Compare the three models' Random-Forest predictions (+ a price-path plot).

---

## Repository layout

```
Eindwerk Syntra VDO/
├── data/raw/            anonymised weekly extract (never modified)
├── src/
│   ├── mekong_delta_price_features.py   PriceFeatureBuilder: targets + engineered features
│   ├── price_model_pipeline.py          PriceModelPipeline: Ridge/Lasso/RF, leakage-safe
│   ├── coverage_analysis.py             CoverageAnalyzer: size the X%-occupancy threshold
│   └── price_comparison.py              compare_rf_predictions across the models
├── notebooks/
│   ├── Price_imitator_EDA.ipynb         exploration (story → methodology)
│   └── Price_imitator.ipynb             the modelling pipeline

```

---

## Setup & run

- **Python 3.13** (system install; no conda needed).
- Install the dependencies:
  ```
  pip install pandas numpy scikit-learn matplotlib seaborn ipywidgets jupyter
  ```
- Place the dataset at `data/raw/` (see above). Run **from the project root** so the
  repo-relative paths resolve.
- Open the notebooks in Jupyter / VS Code and **Run All**, or execute headlessly:
  ```
  python -m jupyter nbconvert --to notebook --execute --inplace notebooks/Price_imitator.ipynb
  ```

> If you edit a `src/` module while a kernel is open, **restart the kernel** (or use
> `%autoreload`) so the new code is picked up.

---

## Headline results (LS-spring test)

All metrics are on the euro scale; the model's value is its **lift over the segment-mean
baseline** (`SeasonalCluster × AccommodationType × RangeType`).

| Model | Best R² | MAPE |
|---|---|---|
| **Imitator** (`InitialPrice`) | **RF 0.51** | 13.8% |
| **Occupancy initializer** (`TargetPrice`) | Ridge/Lasso ~0.49 | ~16% |
| **Lead-time initializer** (`LeadTimePrice`) | RF 0.45 | 17.9% |

(Segment baselines: 0.30 / 0.17 / 0.15 respectively.) Full tables in
[`summary.md`](summary.md) §4.

---

## Key findings

- A **segment mean is a strong baseline**; the models earn their keep mostly in the harder,
  higher-value segments. Season is the dominant price driver, then range tier and campsite.
- **Campsite-aware, hierarchical** last-year imputation (justified by the inter-campsite EDA)
  was the single biggest model gain (imitator RF 0.35 → 0.50).
- **Feature selection, validated by ablation**: dropped noisy last-year repricing-dynamics,
  dropped a redundant `RangeType` one-hot (kept the ordinal), and fixed a silently-zero
  `LastYearEarlyFill` — each step inspected and kept only if it helped.
- The **lead-time target** is denser/less right-censored than the occupancy target.
- Judging whether a recommended price is *better* (not just imitated) needs an **experiment**:
  price is endogenous to demand/quality here, so an offline demand model isn't identifiable.

---

## Limitations

- **LS-spring proof-of-concept** (data, not choice): no other-season rows exist in the 2026
  test year, and a real last-year price exists for only ~40% of options.
- The **X%-occupancy target is weakly supported / right-censored** (half of option-weeks
  never reach 50% fill).
- **Anonymisation** flattens some geography/quality signal and the range ladder.

---

