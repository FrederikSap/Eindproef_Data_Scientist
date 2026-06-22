"""Coverage analysis for choosing the %-booked target threshold (X%).

Goal: pick the highest defensible threshold X% for the "price at X% booked"
target, without half the options never reaching that threshold.

Approach:
1. Build the booking curve per option-arrival-week and take its *maximum* fill.
2. Coverage curve: what share of options reaches each candidate X%?
3. Per group (Season x Region): coverage split out, because a global X% can look
   good on average yet completely censor weak groups.
4. Recommendation: the highest X% at which most groups stay above a coverage floor.

All compute/plot logic lives here; notebooks only import and call.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure

logger = logging.getLogger(__name__)


@dataclass
class Columns:
    """Column names in the dataset (override as needed)."""

    option_id: str = "ReservableOptionId"
    arrival_week: str = "WeekStartDate"
    wba: str = "WeekBeforeArrival"
    cum_booked_nights: str = "CumulativeHistoricalBookedNights"
    capacity: str = "Capacity"
    season: str = "SeasonalCluster"
    region: str = "CampsiteRegion"


class CoverageAnalyzer:
    """Computes fill-coverage to choose the target threshold X%.

    A 'curve' = one (option x arrival-week). The fill of a curve at a given WBA is
    ``cumulative booked nights / capacity``, clipped at 1.0. The *max fill* of a
    curve is the highest fill level it ever reaches (optionally restricted to
    WBA >= min_lead_weeks).

    Args:
        df: Raw booking data.
        columns: Mapping of logical to actual column names.
        min_lead_weeks: Require X% to be reached with at least this many weeks on
            the clock (otherwise a last-minute fill does not count). 0 = no
            requirement.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        columns: Columns | None = None,
        min_lead_weeks: int = 0,
    ) -> None:
        self.cols = columns or Columns()
        self.min_lead_weeks = min_lead_weeks
        self._validate(df)
        self._curves = self._build_curves(df)
        logger.info(
            "%d curves (option x arrival-week) across %d groups.",
            len(self._curves),
            self._curves["group"].nunique(),
        )

    def _validate(self, df: pd.DataFrame) -> None:
        needed = [
            self.cols.option_id, self.cols.arrival_week, self.cols.wba,
            self.cols.cum_booked_nights, self.cols.capacity,
            self.cols.season, self.cols.region,
        ]
        missing = [c for c in needed if c not in df.columns]
        if missing:
            raise KeyError(
                f"Missing columns: {missing}. Provide the correct names via the "
                "Columns config."
            )

    def _build_curves(self, df: pd.DataFrame) -> pd.DataFrame:
        c = self.cols
        work = df[[c.option_id, c.arrival_week, c.wba,
                   c.cum_booked_nights, c.capacity, c.season, c.region]].copy()

        # Rows without a usable numerator/denominator cannot be used.
        before = len(work)
        work = work.dropna(subset=[c.cum_booked_nights, c.capacity])
        work = work[work[c.capacity] > 0]
        dropped = before - len(work)
        if dropped:
            logger.warning("%d rows dropped (no capacity/bookings).", dropped)

        # Fill = booked nights / capacity, clipped at 1.
        work["fill"] = work[c.cum_booked_nights] / work[c.capacity]
        over_one = float((work["fill"] > 1.0).mean())
        if over_one > 0.01:
            logger.warning(
                "%.1f%% of rows have fill > 1.0 -- check the denominator "
                "(capacity). Values are clipped to 1.0.",
                over_one * 100,
            )
        work["fill"] = work["fill"].clip(0.0, 1.0)

        # Possible duplicate rows per (option, week, wba) -- e.g. per market: take max.
        key = [c.option_id, c.arrival_week, c.wba]
        grouped = work.groupby(key, as_index=False).agg(
            fill=("fill", "max"),
            season=(c.season, "first"),
            region=(c.region, "first"),
        )

        # Timing requirement: only count WBAs with enough lead time.
        grouped = grouped[grouped[c.wba] >= self.min_lead_weeks]

        # Max fill per curve (= option x arrival-week).
        curves = grouped.groupby([c.option_id, c.arrival_week], as_index=False).agg(
            max_fill=("fill", "max"),
            season=("season", "first"),
            region=("region", "first"),
        )
        curves["group"] = curves["season"].astype(str) + " | " + curves["region"].astype(str)

        logger.info(
            "Fill distribution (max per curve): median=%.2f, p25=%.2f, p75=%.2f",
            curves["max_fill"].median(),
            curves["max_fill"].quantile(0.25),
            curves["max_fill"].quantile(0.75),
        )
        return curves

    # --- computations -------------------------------------------------------
    def coverage(self, thresholds: np.ndarray) -> pd.Series:
        """Global coverage: share of curves with max_fill >= X, per threshold."""
        return pd.Series(
            {x: float((self._curves["max_fill"] >= x).mean()) for x in thresholds},
            name="coverage",
        )

    def coverage_by_group(self, thresholds: np.ndarray) -> pd.DataFrame:
        """Coverage per Season x Region group (rows = group, columns = X%)."""
        rows = {}
        for group, sub in self._curves.groupby("group"):
            rows[group] = {x: float((sub["max_fill"] >= x).mean()) for x in thresholds}
        out = pd.DataFrame(rows).T
        out.index.name = "group"
        # Sort groups by coverage at the middle threshold (weakest at the bottom).
        mid = thresholds[len(thresholds) // 2]
        return out.sort_values(mid, ascending=False)

    def threshold_for_target_coverage(
        self, season_targets: dict[str, float]
    ) -> pd.DataFrame:
        """Highest %-booked threshold per Season x Region group for a target coverage.

        For a target coverage ``t`` the highest attainable threshold is the
        ``(1 - t)`` quantile of that group's max-fill distribution: the highest X%
        at which at least ``t`` of the curves still reach X%.

        Args:
            season_targets: Target coverage per ``SeasonalCluster`` (e.g.
                ``{"HS": 0.75, "SS": 0.60, "LS": 0.60, "WTR": 0.50}``). Groups whose
                season is not in the mapping are skipped.

        Returns:
            DataFrame with, per group: season, region, target coverage, number of
            curves and the matching ``%booked`` threshold. Sorted by season and region.
        """
        rows = []
        for (season, region), sub in self._curves.groupby(["season", "region"]):
            target = season_targets.get(str(season))
            if target is None:
                continue
            threshold = float(sub["max_fill"].quantile(1.0 - target))
            rows.append({
                "season": season,
                "region": region,
                "target_coverage": target,
                "n_curves": len(sub),
                "pct_booked": threshold,
            })
        out = pd.DataFrame(rows).sort_values(["season", "region"]).reset_index(drop=True)
        return out

    def summary_table(self, thresholds: np.ndarray, coverage_floor: float = 0.8) -> pd.DataFrame:
        """Per candidate X%: global + minimum/median group coverage + #groups below the floor."""
        by_group = self.coverage_by_group(thresholds)
        overall = self.coverage(thresholds)
        rows = []
        for x in thresholds:
            col = by_group[x]
            rows.append({
                "X%": x,
                "global_coverage": overall[x],
                "min_group_coverage": col.min(),
                "median_group_coverage": col.median(),
                "groups_below_floor": int((col < coverage_floor).sum()),
            })
        return pd.DataFrame(rows).set_index("X%")

    def recommend(self, thresholds: np.ndarray, coverage_floor: float = 0.8) -> float:
        """Highest X% at which ALL groups stay above the coverage floor.

        Falls back to the highest X% with the best minimum-group coverage when no
        threshold satisfies the floor.
        """
        table = self.summary_table(thresholds, coverage_floor)
        ok = table[table["groups_below_floor"] == 0]
        if not ok.empty:
            choice = float(ok.index.max())
            logger.info("Recommended X%% = %.0f%% (all groups >= %.0f%% coverage).",
                        choice * 100, coverage_floor * 100)
            return choice
        choice = float(table["min_group_coverage"].idxmax())
        logger.warning(
            "No threshold keeps all groups above %.0f%%. Best compromise: X%%=%.0f%%.",
            coverage_floor * 100, choice * 100,
        )
        return choice

    # --- plots --------------------------------------------------------------
    def plot_max_fill_distribution(self) -> Figure:
        """Histogram of the max fill per curve (what is achievable?)."""
        sns.set_theme(style="whitegrid")
        fig, ax = plt.subplots(figsize=(7, 4))
        sns.histplot(self._curves["max_fill"], bins=25, ax=ax)
        ax.axvline(self._curves["max_fill"].median(), color="crimson", ls="--",
                   label=f"median = {self._curves['max_fill'].median():.0%}")
        ax.set_xlabel("Max fill per option-arrival-week")
        ax.set_ylabel("Number of curves")
        ax.set_title("What is achievable? Distribution of the maximum occupancy")
        ax.legend()
        fig.tight_layout()
        return fig

    def plot_coverage_curve(self, thresholds: np.ndarray, coverage_floor: float = 0.8) -> Figure:
        """Global coverage curve with the coverage floor and the recommended X%."""
        sns.set_theme(style="whitegrid")
        cov = self.coverage(thresholds)
        rec = self.recommend(thresholds, coverage_floor)

        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.plot(cov.index * 100, cov.values * 100, marker="o")
        ax.axhline(coverage_floor * 100, color="darkorange", ls="--",
                   label=f"coverage floor ({coverage_floor:.0%})")
        ax.axvline(rec * 100, color="crimson", ls="--",
                   label=f"recommended X% = {rec:.0%}")
        ax.set_xlabel("Candidate threshold X% (% booked)")
        ax.set_ylabel("Coverage: % of options reaching X%")
        ax.set_title("Coverage curve: how many options reach each threshold?")
        ax.legend()
        fig.tight_layout()
        return fig

    def plot_group_heatmap(self, thresholds: np.ndarray) -> Figure:
        """Heatmap of coverage per Season x Region group across all candidate X%."""
        sns.set_theme(style="white")
        data = self.coverage_by_group(thresholds)
        data.columns = [f"{int(x * 100)}%" for x in thresholds]

        fig, ax = plt.subplots(figsize=(max(8, len(thresholds) * 0.6), 0.6 * len(data) + 2))
        sns.heatmap(data, annot=True, fmt=".0%", cmap="RdYlGn", vmin=0, vmax=1,
                    cbar_kws={"label": "coverage"}, linewidths=0.5, ax=ax)
        ax.set_xlabel("Candidate threshold X% (% booked)")
        ax.set_ylabel("Season | Region")
        ax.set_title("Coverage per group -- pick X% where no row turns red")
        fig.tight_layout()
        return fig
