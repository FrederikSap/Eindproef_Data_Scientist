"""Train/evaluate Ridge, Lasso and Random Forest price models on an option-week table.

:class:`PriceModelPipeline` is the reusable engine behind the clean price-imitator
notebook. Given a feature table from
:mod:`src.mekong_delta_price_features` and a target column (``InitialPrice`` or
``TargetPrice``), it runs a single, leakage-safe flow:

1. split by arrival-week date masks (train season vs. held-out season) and drop
   rows with a missing or non-positive target,
2. pool rare ``CampsiteCode`` levels (learned on train) into ``"Other"``,
3. add a ``has_last_year`` flag and impute the sparse last-year columns within a
   ``SeasonalCluster × RangeType`` segment (train-learned, global fallback),
4. one-hot encode categoricals (``handle_unknown="ignore"``) and median-impute
   numerics inside a single :class:`~sklearn.compose.ColumnTransformer`,
5. optionally fit on ``log1p(price)`` and back-transform predictions,
6. fit ``RidgeCV``, ``LassoCV`` (both on the standardised design) and a
   ``TimeSeriesSplit``-tuned ``RandomForestRegressor`` (on the raw design),
7. report RMSE / MAE / R² / MAPE, global + segment baselines, and a
   per-``SeasonalCluster`` RMSE table.

The fitted models, the aligned design matrices, and the test metadata (including
``ReservableOptionId``) are kept on a :class:`PriceModelResult` so a later step
can compare two pipelines' Ridge predictions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LassoCV, RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


@dataclass
class PipelineConfig:
    """Configuration for :class:`PriceModelPipeline`.

    Attributes:
        target: Target price column to predict.
        last_year_feature: Prior-season price feature paired with ``target``.
        categorical_columns: Features expanded by one-hot encoding.
        numeric_columns: Features median-imputed and standardised (for the linear
            models). ``last_year_feature`` is prepended automatically.
        bool_columns: Boolean attributes cast to ``float`` before modelling.
        last_year_columns: Sparse prior-season columns imputed within a segment.
        last_year_segment: Segment used for the train-learned last-year imputation.
        campsite_column: High-cardinality id pooled into ``"Other"``.
        campsite_min_options: Minimum train frequency a ``campsite_column`` level
            needs to survive pooling.
        segment_columns: Columns defining the baseline comparison segment.
        season_column: Seasonal-cluster column for per-segment reporting.
        week_column: Arrival-week date column the split masks act on.
        option_id_column: Identifier carried onto the test predictions.
        train_start/train_end: Inclusive/exclusive train window (``YYYY-MM-DD``).
        test_start/test_end: Inclusive test window.
        log_target: Fit on ``log1p(target)`` and back-transform (stabilises the
            right-skewed price variance, improves MAPE).
        alphas: Ridge regularisation grid.
        lasso_alphas: Optional Lasso grid; ``None`` lets ``LassoCV`` pick its own.
        cv_splits: Number of forward-chaining ``TimeSeriesSplit`` folds.
        rf_param_grid: Random-forest grid searched by :class:`GridSearchCV`.
        rf_estimators: Number of trees in the forest.
        random_state: Seed for the forest.
    """

    target: str = "InitialPrice"
    last_year_feature: str = "InitialPriceLastYear"
    categorical_columns: list[str] = field(
        default_factory=lambda: [
            "CampsiteCodeGrouped",
            "SeasonalCluster",
            "AccommodationType",
            # RangeType (anonymised one-hot) dropped after an ablation — RangeOrdinal
            # carries the same quality-tier signal with one column; the linear models
            # improved and RF was unchanged. RangeType is still used by the imputation
            # and baseline segments below, just not as a model feature.
            "CampsiteType",
            "DeckingType",
            "BrandGroupCode",
        ]
    )
    numeric_columns: list[str] = field(
        default_factory=lambda: [
            "RangeOrdinal",
            "SleepsNum",
            "IsoWeek",
            "ArrivalMonth",
            "Bedrooms",
            "Bathrooms",
            "Capacity",
            "Airco",
            "TV",
            "is_special_period",
            "has_last_year",
            "LastYearMaxFill",
            "LastYearEarlyFill",
        ]
    )
    bool_columns: list[str] = field(default_factory=lambda: ["Airco", "TV"])
    last_year_columns: list[str] = field(
        default_factory=lambda: ["LastYearMaxFill", "LastYearEarlyFill"]
    )
    last_year_segment: list[str] = field(
        default_factory=lambda: ["SeasonalCluster", "RangeType", "CampsiteCode"]
    )
    campsite_column: str = "CampsiteCode"
    campsite_min_options: int = 30
    train_subset_column: Optional[str] = None
    segment_columns: list[str] = field(
        default_factory=lambda: ["SeasonalCluster", "AccommodationType", "RangeType"]
    )
    season_column: str = "SeasonalCluster"
    week_column: str = "WeekStartDate"
    option_id_column: str = "ReservableOptionId"
    train_start: str = "2025-01-01"
    train_end: str = "2025-06-08"
    test_start: str = "2026-01-01"
    test_end: str = "2026-06-06"
    log_target: bool = True
    alphas: np.ndarray = field(default_factory=lambda: np.logspace(-3, 3, 25))
    lasso_alphas: np.ndarray = field(default_factory=lambda: np.logspace(-3, 1, 50))
    cv_splits: int = 5
    rf_param_grid: dict[str, list] = field(
        default_factory=lambda: {
            "max_depth": [None, 10, 20],
            "min_samples_leaf": [1, 5, 10],
        }
    )
    rf_estimators: int = 300
    random_state: int = 42


@dataclass
class PriceModelResult:
    """Fitted models, predictions and diagnostics from one pipeline run.

    Attributes:
        config: The configuration used.
        ridge: Fitted ``RidgeCV`` model.
        lasso: Fitted ``LassoCV`` model.
        random_forest: Fitted, CV-tuned random forest.
        preprocessor: Fitted ``ColumnTransformer`` (impute + one-hot).
        scaler: Scaler fitted on the encoded train design (linear-model input).
        x_train: Encoded, unscaled train design matrix (named columns).
        x_test: Encoded, unscaled test design matrix.
        y_train: Train target vector (original scale).
        y_test: Test target vector (original scale).
        test_meta: Test-row metadata (option id, season, segment columns, target).
        metrics: Per-model overall metrics (Ridge / Lasso / RF / baselines).
        ridge_predictions: Ridge predictions on the test set (original scale).
        lasso_predictions: Lasso predictions on the test set (original scale).
        rf_predictions: Random-forest predictions on the test set (original scale).
        baseline_segment: Segment-mean baseline predictions on the test set.
        per_cluster_rmse: RMSE per ``SeasonalCluster`` per model.
        single_season: The sole season code if the test window holds only one
            (an honesty flag), else ``None``.
    """

    config: PipelineConfig
    ridge: RidgeCV
    lasso: LassoCV
    random_forest: RandomForestRegressor
    preprocessor: ColumnTransformer
    scaler: StandardScaler
    x_train: pd.DataFrame
    x_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    test_meta: pd.DataFrame
    metrics: pd.DataFrame
    ridge_predictions: np.ndarray
    lasso_predictions: np.ndarray
    rf_predictions: np.ndarray
    baseline_segment: np.ndarray
    per_cluster_rmse: pd.DataFrame
    single_season: Optional[str] = None


def regression_report(
    y_true: np.ndarray, y_pred: np.ndarray, label: str = ""
) -> dict[str, float]:
    """Compute and print RMSE, MAE, R² and MAPE for a prediction vector.

    MAPE is computed only over non-zero truths to avoid divide-by-zero.

    Args:
        y_true: Ground-truth values.
        y_pred: Predicted values.
        label: Row label printed alongside the metrics.

    Returns:
        Mapping with ``rmse``, ``mae``, ``r2`` and ``mape`` keys.
    """
    y_true = np.asarray(y_true, dtype="float64").ravel()
    y_pred = np.asarray(y_pred, dtype="float64").ravel()
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    nonzero = y_true != 0
    mape = float(
        np.mean(np.abs((y_true[nonzero] - y_pred[nonzero]) / y_true[nonzero])) * 100
    )
    print(f"{label:>16} | RMSE {rmse:8.2f} | MAE {mae:8.2f} | R2 {r2:6.3f} | MAPE {mape:5.1f}%")
    return {"rmse": rmse, "mae": mae, "r2": r2, "mape": mape}


class PriceModelPipeline:
    """Runs the Ridge + Lasso + Random Forest price-modelling flow for one target.

    Args:
        config: Pipeline configuration. Defaults to an ``InitialPrice`` setup.
    """

    def __init__(self, config: Optional[PipelineConfig] = None) -> None:
        self.config = config or PipelineConfig()

    # --- public API ---------------------------------------------------------
    def run(self, table: pd.DataFrame) -> PriceModelResult:
        """Train, evaluate and package the models for ``config.target``.

        Args:
            table: Option-week feature table carrying the target, its last-year
                feature, and the configured feature columns.

        Returns:
            A populated :class:`PriceModelResult`.
        """
        cfg = self.config
        train_df, test_df, single_season = self._split(table)
        train_df, test_df = self._prepare(train_df, test_df)

        y_train = train_df[cfg.target].copy()
        y_test = test_df[cfg.target].copy()
        # Fit on log-price when configured; predictions are back-transformed so
        # every metric below is reported on the original euro scale.
        inv = np.expm1 if cfg.log_target else (lambda a: a)
        y_fit = np.log1p(y_train) if cfg.log_target else y_train

        x_train, x_test, preprocessor = self._build_design(train_df, test_df)
        scaler, x_train_scaled, x_test_scaled = self._scale(x_train, x_test)
        cv = TimeSeriesSplit(n_splits=cfg.cv_splits)

        ridge = RidgeCV(alphas=cfg.alphas)
        ridge.fit(x_train_scaled, y_fit)
        ridge_pred = inv(ridge.predict(x_test_scaled))

        lasso = LassoCV(
            alphas=cfg.lasso_alphas, cv=cv, max_iter=50_000, random_state=cfg.random_state
        )
        lasso.fit(x_train_scaled, np.asarray(y_fit).ravel())
        lasso_pred = inv(lasso.predict(x_test_scaled))

        forest = self._fit_forest(x_train, y_fit, cv)
        rf_pred = inv(forest.predict(x_test))

        baseline_segment = self._segment_baseline(train_df, test_df)
        metrics = self._collect_metrics(
            y_test, ridge_pred, lasso_pred, rf_pred, y_train, baseline_segment
        )
        test_meta = self._build_test_meta(test_df)
        per_cluster = self._per_cluster_rmse(
            test_meta, y_test, baseline_segment, ridge_pred, lasso_pred, rf_pred
        )

        return PriceModelResult(
            config=cfg,
            ridge=ridge,
            lasso=lasso,
            random_forest=forest,
            preprocessor=preprocessor,
            scaler=scaler,
            x_train=x_train,
            x_test=x_test,
            y_train=y_train,
            y_test=y_test,
            test_meta=test_meta,
            metrics=metrics,
            ridge_predictions=np.asarray(ridge_pred).ravel(),
            lasso_predictions=np.asarray(lasso_pred).ravel(),
            rf_predictions=np.asarray(rf_pred).ravel(),
            baseline_segment=baseline_segment,
            per_cluster_rmse=per_cluster,
            single_season=single_season,
        )

    def ridge_coefficients(self, result: PriceModelResult, top_n: int = 15) -> pd.Series:
        """Return the largest-magnitude Ridge coefficients (scaled features)."""
        coef = pd.Series(
            result.ridge.coef_.ravel(), index=result.x_train.columns
        ).sort_values(key=np.abs, ascending=False)
        return coef.head(top_n)

    def lasso_coefficients(self, result: PriceModelResult, top_n: int = 15) -> pd.Series:
        """Return the largest-magnitude *non-zero* Lasso coefficients."""
        coef = pd.Series(result.lasso.coef_.ravel(), index=result.x_train.columns)
        coef = coef[coef != 0].sort_values(key=np.abs, ascending=False)
        return coef.head(top_n)

    def rf_importances(self, result: PriceModelResult, top_n: int = 15) -> pd.Series:
        """Return the largest random-forest feature importances."""
        importances = pd.Series(
            result.random_forest.feature_importances_, index=result.x_train.columns
        ).sort_values(ascending=False)
        return importances.head(top_n)

    # --- internals ----------------------------------------------------------
    def _split(
        self, table: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame, Optional[str]]:
        """Date-mask split; drop missing/non-positive targets; sort train by time."""
        cfg = self.config
        week = pd.to_datetime(table[cfg.week_column])
        train_mask = (week >= cfg.train_start) & (week < cfg.train_end)
        test_mask = (week >= cfg.test_start) & (week <= cfg.test_end)

        def _clean(mask: pd.Series) -> pd.DataFrame:
            sub = table.loc[mask].dropna(subset=[cfg.target]).copy()
            return sub[sub[cfg.target] > 0]

        # Sort train by arrival week so TimeSeriesSplit folds are forward-chaining.
        train_df = _clean(train_mask).sort_values(cfg.week_column).reset_index(drop=True)
        test_df = _clean(test_mask).reset_index(drop=True)

        # Optionally restrict TRAINING to a subset (e.g. the on-target rows) while
        # leaving the held-out test set complete for diagnostics.
        if cfg.train_subset_column and cfg.train_subset_column in train_df.columns:
            before = len(train_df)
            train_df = train_df[train_df[cfg.train_subset_column].astype(bool)].reset_index(
                drop=True
            )
            print(
                f"train restricted to '{cfg.train_subset_column}': {len(train_df)} of "
                f"{before} rows"
            )

        seasons = sorted(test_df[cfg.season_column].dropna().unique())
        single_season = seasons[0] if len(seasons) == 1 else None

        print(f"train rows {len(train_df):,} | test rows {len(test_df):,}")
        print(
            f"train WSD {week[train_mask].min().date()} -> {week[train_mask].max().date()}"
        )
        print(
            f"test  WSD {week[test_mask].min().date()} -> {week[test_mask].max().date()}"
        )
        print(
            f"y_train mean {train_df[cfg.target].mean():.1f}  "
            f"y_test mean {test_df[cfg.target].mean():.1f}"
        )
        if single_season is not None:
            print(
                f"!! test window holds ONE season ('{single_season}') — metrics are a "
                f"'{single_season}'-only proof-of-concept, not a general price model."
            )
        return train_df, test_df, single_season

    def _prepare(
        self, train_df: pd.DataFrame, test_df: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Bool-cast, pool rare campsites, flag and segment-impute last-year cols."""
        cfg = self.config
        train_df = train_df.copy()
        test_df = test_df.copy()

        for col in cfg.bool_columns:
            for frame in (train_df, test_df):
                if col in frame.columns:
                    frame[col] = frame[col].astype("float64")

        # Pool rare campsite levels using the TRAIN frequency only.
        if cfg.campsite_column in train_df.columns:
            counts = train_df[cfg.campsite_column].value_counts()
            keep = set(counts[counts >= cfg.campsite_min_options].index)
            for frame in (train_df, test_df):
                frame["CampsiteCodeGrouped"] = frame[cfg.campsite_column].where(
                    frame[cfg.campsite_column].isin(keep), "Other"
                )

        # Missingness flag captured BEFORE any imputation.
        for frame in (train_df, test_df):
            frame["has_last_year"] = frame[cfg.last_year_feature].notna().astype(int)

        ly_cols = [cfg.last_year_feature, *cfg.last_year_columns]
        ly_cols = [c for c in ly_cols if c in train_df.columns]
        seg = [c for c in cfg.last_year_segment if c in train_df.columns]
        train_df, test_df = self._segment_impute(train_df, test_df, ly_cols, seg)
        return train_df, test_df

    def _segment_impute(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        columns: list[str],
        segment: list[str],
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Fill missing ``columns`` with a **hierarchical** train-median fallback.

        For each column, missing values are filled by trying progressively coarser
        segment medians and stopping at the first that resolves —
        e.g. ``[Season, Range, Campsite] → [Season, Range] → [Season] → global``.
        This keeps each imputed value in the most specific sensible price band (the
        campsite's own level, per EDA §7) while staying robust where a fine cell is
        empty. All medians are learned on the **train** split only.
        """
        levels = [segment[:k] for k in range(len(segment), 0, -1)]
        for col in columns:
            global_median = float(train_df[col].median())
            for frame in (train_df, test_df):
                vals = frame[col].to_numpy(dtype="float64").copy()
                need = np.isnan(vals)
                for seg in levels:
                    if not need.any():
                        break
                    median = train_df.groupby(seg)[col].median()
                    mapped = np.asarray(
                        frame.set_index(seg).index.map(median), dtype="float64"
                    )
                    fill = need & ~np.isnan(mapped)
                    vals[fill] = mapped[fill]
                    need = np.isnan(vals)
                vals[np.isnan(vals)] = global_median
                frame[col] = vals
        return train_df, test_df

    def _build_design(
        self, train_df: pd.DataFrame, test_df: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame, ColumnTransformer]:
        """One-hot + median-impute via a leak-safe ``ColumnTransformer``.

        ``handle_unknown="ignore"`` means a campsite/category seen only at test
        time encodes to all-zeros instead of raising or needing a manual
        column-reindex.
        """
        cfg = self.config
        cat = [c for c in cfg.categorical_columns if c in train_df.columns]
        num = [cfg.last_year_feature, *cfg.numeric_columns]
        num = [c for c in dict.fromkeys(num) if c in train_df.columns]

        categorical = Pipeline(
            [
                ("impute", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]
        )
        pre = ColumnTransformer(
            [
                ("cat", categorical, cat),
                ("num", SimpleImputer(strategy="median"), num),
            ],
            remainder="drop",
            verbose_feature_names_out=False,
        )

        x_train = pre.fit_transform(train_df)
        x_test = pre.transform(test_df)
        names = list(pre.get_feature_names_out())
        x_train = pd.DataFrame(x_train, columns=names, index=train_df.index)
        x_test = pd.DataFrame(x_test, columns=names, index=test_df.index)
        return x_train, x_test, pre

    def _scale(
        self, x_train: pd.DataFrame, x_test: pd.DataFrame
    ) -> tuple[StandardScaler, pd.DataFrame, pd.DataFrame]:
        """Standardise the full encoded design (linear-model input)."""
        scaler = StandardScaler()
        x_train_scaled = pd.DataFrame(
            scaler.fit_transform(x_train), columns=x_train.columns, index=x_train.index
        )
        x_test_scaled = pd.DataFrame(
            scaler.transform(x_test), columns=x_test.columns, index=x_test.index
        )
        return scaler, x_train_scaled, x_test_scaled

    def _fit_forest(
        self, x_train: pd.DataFrame, y_train: pd.Series, cv: TimeSeriesSplit
    ) -> RandomForestRegressor:
        """CV-tune and fit the random forest on the unscaled design (time-aware CV)."""
        cfg = self.config
        base = RandomForestRegressor(
            n_estimators=cfg.rf_estimators,
            random_state=cfg.random_state,
            n_jobs=-1,
        )
        search = GridSearchCV(
            base, cfg.rf_param_grid, cv=cv, scoring="neg_root_mean_squared_error"
        )
        search.fit(x_train, np.asarray(y_train).ravel())
        print(f"random forest best params: {search.best_params_}")
        return search.best_estimator_

    def _segment_baseline(
        self, train_df: pd.DataFrame, test_df: pd.DataFrame
    ) -> np.ndarray:
        """Predict the train segment-mean target for each test row."""
        cfg = self.config
        seg_means = train_df.groupby(cfg.segment_columns)[cfg.target].mean()
        global_mean = float(train_df[cfg.target].mean())
        mapped = test_df.set_index(cfg.segment_columns).index.map(seg_means)
        baseline = mapped.to_numpy(dtype="float64")
        return np.where(np.isnan(baseline), global_mean, baseline)

    def _collect_metrics(
        self,
        y_test: pd.Series,
        ridge_pred: np.ndarray,
        lasso_pred: np.ndarray,
        rf_pred: np.ndarray,
        y_train: pd.Series,
        baseline_segment: np.ndarray,
    ) -> pd.DataFrame:
        """Print and tabulate overall metrics for every model and baseline."""
        rows = {
            "Baseline-mean": regression_report(
                y_test, np.full(len(y_test), float(y_train.mean())), "Baseline-mean"
            ),
            "Baseline-segment": regression_report(
                y_test, baseline_segment, "Baseline-segment"
            ),
            "Ridge": regression_report(y_test, ridge_pred, "Ridge"),
            "Lasso": regression_report(y_test, lasso_pred, "Lasso"),
            "RandomForest": regression_report(y_test, rf_pred, "RandomForest"),
        }
        return pd.DataFrame(rows).T

    def _build_test_meta(self, test_df: pd.DataFrame) -> pd.DataFrame:
        """Carry identifiers and segment columns alongside the test target."""
        cfg = self.config
        meta_cols = [cfg.option_id_column, cfg.week_column, cfg.season_column]
        meta_cols += [c for c in cfg.segment_columns if c not in meta_cols]
        meta_cols = [c for c in dict.fromkeys(meta_cols) if c in test_df.columns]
        meta = test_df[meta_cols].reset_index(drop=True).copy()
        meta[cfg.target] = test_df[cfg.target].to_numpy()
        return meta

    def _per_cluster_rmse(
        self,
        test_meta: pd.DataFrame,
        y_test: pd.Series,
        baseline_segment: np.ndarray,
        ridge_pred: np.ndarray,
        lasso_pred: np.ndarray,
        rf_pred: np.ndarray,
    ) -> pd.DataFrame:
        """RMSE per ``SeasonalCluster`` for baseline, Ridge, Lasso and forest."""
        cfg = self.config
        frame = pd.DataFrame(
            {
                cfg.season_column: test_meta[cfg.season_column].to_numpy(),
                "y_true": np.asarray(y_test).ravel(),
                "Baseline-segment": np.asarray(baseline_segment).ravel(),
                "Ridge": np.asarray(ridge_pred).ravel(),
                "Lasso": np.asarray(lasso_pred).ravel(),
                "RandomForest": np.asarray(rf_pred).ravel(),
            }
        )

        def _rmse(group: pd.DataFrame) -> pd.Series:
            cols = ["Baseline-segment", "Ridge", "Lasso", "RandomForest"]
            out = {"n": len(group)}
            for col in cols:
                out[col] = mean_squared_error(group["y_true"], group[col]) ** 0.5
            return pd.Series(out)

        return (
            frame.groupby(cfg.season_column).apply(_rmse, include_groups=False).round(2)
        )
