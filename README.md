# Price Imitator — Opening-Price Model for ECG

Predicts an appropriate **opening price** for a reservable camping accommodation from its
characteristics and historical demand. The model learns to *imitate* the price European
Camping Group (ECG) has historically opened the booking window with, given comparable
options in the same season.

Two notebooks make up the work:

| Notebook | Purpose |
|----------|---------|
| [`Price_imitator_EDA.ipynb`](../notebooks/Price_imitator_EDA.ipynb) | Exploratory analysis that motivates every modelling choice. |
| [`Price_imitator_Model.ipynb`](../notebooks/Price_imitator_Model.ipynb) | Feature engineering, target construction, train/test split, baselines, and two models. |

Code can be found on github.com/FrederikSap/Eindproef_Data_Scientist.git
---

## Data

- **Source:** `data/raw/French_Riviera_Multiple_campsites_datasets_capacity_over_200_anonymised.csv` -- Not included because it contains sensitive bookingsdata
- **Grain:** one row per `ReservableOptionId × WeekBeforeArrival × WeekStartDate × MarketGroupCode`.
- **Key relationships established in the EDA:**
  - `ReservableOptionId` maps **one-to-one** onto an arrival week (`WeekStartDate`).
  - Each option carries **4 market groups**, each with its own bookings; these must be
    aggregated up to a single option-week record before modelling.

---

## EDA — what it checks and why

The EDA notebook is ordered so each section justifies a later modelling decision.

1. **Load + shape** — read the CSV, parse `WeekStartDate`, derive `Year`.
2. **Grain validation** — confirm `ReservableOptionId` uniquely identifies an arrival week.
3. **Market-group structure** — show 4 market groups per option and aggregate bookings to
   the option level.
4. **Data-quality checks**
   - *Monotonicity* — `CumulativeHistoricalBookedNights` never decreases as arrival approaches.
   - *Overbooking* — flag rows where `TotalBookedNights > Capacity`.
   - *LastYear missingness* — separate genuine prior-year data from placeholder fills.
5. **Feature diagnostics**
   - *Categorical cardinality* — flag high-cardinality fields (e.g. `CampsiteCode`) that
     blow up one-hot width and destabilise Ridge.
   - *Boolean constancy within `AccoTypeRangeCode`* — drop attributes fully determined by
     the type|range encoding; keep those that carry independent signal.
6. **Seasonal structure**
   - `season_pivot` confirms every `SeasonalCluster` appears in **both** season years, so
     the 2024→2025 split and the segment baseline have cross-year support.
   - Capacity-weighted booking-fill per campsite (interactive campsite dropdown) shows the
     fill-rate shape differs sharply by season → **targets must be defined per season.**
7. **Target definition** — see below.

---

## Target definition

Both candidates collapse the option × horizon panel to **one target per reservable option**.

- **A — Demand-weighted opening price (used by the model).**
  Collapse the 4 market groups into a single demand-weighted price per option × horizon
  (weight = `CumulativeHistoricalBookedNights`), then read the price at the **opening of the
  booking window** (largest `WeekBeforeArrival` with real demand). A *price imitator*:
  reproduces the historical opening price. Well-defined for every option, no threshold
  dials, dense and leakage-free.

- **B — `TargetLabeler` good-price subset (future work).**
  Label each option-week `underpriced` / `on_target` / `overpriced` from how it sold, and
  train only on `on_target` rows. A *price initializer*: learns only from prices that
  produced good outcomes. Discards most rows and depends on sell-out / occupancy
  assumptions.

The model uses **A**. **B** is the natural next step once the goal shifts from replicating
to *correcting* historical pricing. (see last point continuation)

---

## Model pipeline

1. **Feature engineering** (`Price_imitator_Model.ipynb`)
   - `Year` from `WeekStartDate` for time-based splitting.
   - `Bedrooms` extracted to integer.
   - `AccoTypeRangeCode` split into `AccommodationType` + `RangeType`.
   - `is_special_period` binary flag from `SpecialPeriodCode`.
2. **Target construction** — demand-weighted `InitialPrice` per option (candidate A above),
   collapsed to one row per `ReservableOptionId`.
3. **Train/test split** — **2024 train / 2025 test** (never random across arrival weeks).
    group campsites with low unique ReservableOptionId in 'other'
4. **Missing values** — categorical columns (`Airco`, `TV`, `DeckingType`, `Bathrooms`)
   imputed with the mode, fitted on train and test independently.
5. **Encoding** — one-hot encode categoricals; `x_test` reindexed onto `x_train` columns
   (`fill_value=0`) to keep the feature space aligned and leakage-free.
6. **Scaling** — `StandardScaler` on numeric columns for Ridge only.

### Features

`BrandGroupCode`, `CampsiteCode`, `is_special_period`, `SeasonalCluster`, `CampsiteType`,
`AccommodationType`, `RangeType`, `Airco`, `Bedrooms`, `DeckingType`, `Bathrooms`, `TV`,
`Capacity`.

### Models

- **Baselines** — global train-mean price; segment train-mean price
  (`SeasonalCluster × AccommodationType × RangeType`, unseen segments → global mean).
- **Ridge regression** — `RidgeCV` picks `alpha` by CV on train only; coefficients give
  interpretable price drivers.
- **Random Forest** — `GridSearchCV` over `max_depth` and `min_samples_leaf`; feature
  importances show reliance.

---

## Evaluation

- **Primary metric:** RMSE, reported overall **and per `SeasonalCluster`**.
- Per-cluster RMSE compares Ridge and Random Forest against the segment baseline.
- **Finding:** no single model wins everywhere. Random Forest is the best all-round choice
  (beats the baseline in 4/5 tiers), but Ridge is materially better in the two important
  tiers (HS, S1).

---
## Continuation

- Goal is to move from price-imitation to price-initialization-model (see EDA point 7B)
  Due to job-hunting priorities this is not ready, goal is still to have it available at presentation
