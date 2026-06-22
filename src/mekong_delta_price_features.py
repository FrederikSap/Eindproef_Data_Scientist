"""Derive opening (initial) and X%-occupancy (target) prices from weekly data.

The French-Riviera weekly extract carries no explicit opening or target price; it
only records ``AverageDiscPriceForWeekBeforeArrival`` (and its last-year twin) at
every booking horizon. :class:`PriceFeatureBuilder` reconstructs the price
targets the imitator models need from those per-horizon snapshots:

- ``InitialPrice`` — the discounted price at the *earliest* snapshot of the
  current season (the highest ``WeekBeforeArrival`` that has a price), i.e. the
  opening price a guest first saw.
- ``InitialPriceLastYear`` — the same idea on the last-year price series,
  resolved independently (its own earliest non-null horizon).
- ``TargetPrice`` — the price at the first horizon (scanning early → late) where
  cumulative occupancy reaches the season's ``%`` threshold.
- ``TargetPriceLastYear`` — the same threshold rule on the last-year series.

The raw file has several ``MarketGroupCode`` rows per option-horizon; these are
collapsed to one row per ``(ReservableOptionId, WeekStartDate, WeekBeforeArrival)``
(booked nights summed, prices averaged, static attributes taken first) before any
price is derived. All logic lives here so notebooks only import and call.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class PriceFeatureColumns:
    """Column names used by :class:`PriceFeatureBuilder`.

    Attributes:
        option_id: Unique reservable-option identifier.
        arrival_week: Arrival-week start date.
        wba: Booking-horizon countdown (higher = earlier / further out).
        price: Current-season average discounted price at this horizon.
        price_last_year: Prior-season average discounted price at this horizon.
        cum_booked: Cumulative historical booked nights up to this horizon.
        cum_booked_last_year: Same cumulative measure for last year only.
        capacity: Current-season capacity (occupancy denominator).
        capacity_last_year: Prior-season capacity; falls back to ``capacity``.
        season: Seasonal-cluster column whose value selects the ``%`` threshold.
        acco_type_range: Combined ``"MH | Ultimate"`` style code to split.
        special_period: Promotional-period code; non-null becomes a flag.
        market_group: Market-group column that drives the duplicate rows.
    """

    option_id: str = "ReservableOptionId"
    arrival_week: str = "WeekStartDate"
    arrival_month: str = "ArrivalMonth"
    wba: str = "WeekBeforeArrival"
    price: str = "AverageDiscPriceForWeekBeforeArrival"
    price_last_year: str = "AverageDiscPriceForWeekBeforeArrivalLastYear"
    cum_booked: str = "CumulativeHistoricalBookedNights"
    cum_booked_last_year: str = "CumulativeHistoricalBookedNightsLastYear"
    capacity: str = "Capacity"
    capacity_last_year: str = "CapacityLastYear"
    season: str = "SeasonalCluster"
    acco_type_range: str = "AccoTypeRangeCode"
    special_period: str = "SpecialPeriodCode"
    market_group: str = "MarketGroupCode"


# Ordered accommodation-range tiers, low → high quality. Used to encode
# ``RangeType`` as a single ordinal feature (``RangeOrdinal``) instead of a wide
# one-hot block — a monotone price ladder the models can lean on directly.
RANGE_ORDER: dict[str, int] = {
    "Budget": 0,
    "Budget+": 1,
    "Comfort": 2,
    "Comfort+": 3,
    "Premium": 4,
    "Premium+": 5,
    "Ultimate": 6,
}


# Per-season lead time (``WeekBeforeArrival``) at which demand lifts off — the
# anchor for the lead-time price target. Derived from the EDA booking curves
# (mean occupancy first becomes non-trivial ~WBA 30 for HS/S1/S2, ~20 for LS/WTR).
LEAD_BY_SEASON: dict[str, int] = {"HS": 30, "S1": 30, "S2": 30, "LS": 20, "WTR": 20}


# Static accommodation/campsite attributes carried unchanged onto the
# one-row-per-option-week feature table (first value within the option-week).
STATIC_ATTRIBUTES: list[str] = [
    "BrandGroupCode",
    "CampsiteCode",
    "ArrivalMonth",
    "CampsiteType",
    "CampsiteCluster",
    "CampsiteCountry",
    "CampsiteRegion",
    "AccoKindCode",
    "AccommodationType",
    "AccommodationRange",
    "Airco",
    "HotTub",
    "Tropical",
    "Roof",
    "Kitchen",
    "DeckingExtras",
    "DeckingType",
    "Bedrooms",
    "Bathrooms",
    "Sleeps",
    "TV",
]


class PriceFeatureBuilder:
    """Builds an option-week feature table with derived price targets.

    Args:
        columns: Column-name mapping. Defaults to the French-Riviera schema.
        static_attributes: Static columns to carry onto the option-week table.
            Missing columns are skipped. Defaults to :data:`STATIC_ATTRIBUTES`.

    The typical flow is :meth:`build_option_week_table` (once), then
    :meth:`add_initial_prices` and :meth:`add_target_prices` to attach the
    derived target columns.
    """

    def __init__(
        self,
        columns: Optional[PriceFeatureColumns] = None,
        static_attributes: Optional[list[str]] = None,
    ) -> None:
        self.cols = columns or PriceFeatureColumns()
        self._static_attributes = (
            STATIC_ATTRIBUTES if static_attributes is None else static_attributes
        )

    # --- aggregation --------------------------------------------------------
    def collapse_market_rows(self, df: pd.DataFrame) -> pd.DataFrame:
        """Collapse market-group duplicates to one row per option-horizon.

        Booked-night and price columns are aggregated across the market rows
        that share a ``(option, arrival_week, wba)`` key; static attributes are
        taken from the first row.

        Args:
            df: Raw weekly extract.

        Returns:
            Frame with a unique ``(option, arrival_week, wba)`` grain.
        """
        c = self.cols
        df = df.copy()
        for col in (c.price, c.price_last_year):
            if col in df.columns:
                df[col] = df[col].where(df[col] > 0)
        
        key = [c.option_id, c.arrival_week, c.wba]

        agg_spec: dict[str, tuple[str, str]] = {}
        for col, how in (
            (c.price, "mean"),
            (c.price_last_year, "max"),  # max to prefer a price over no price (NaN) when averaging would give NaN
            (c.cum_booked, "sum"),
            (c.cum_booked_last_year, "sum"),
            (c.capacity, "first"),
            (c.capacity_last_year, "first"),
            (c.season, "first"),
            (c.acco_type_range, "first"),
            (c.special_period, "first"),
        ):
            if col in df.columns:
                agg_spec[col] = (col, how)
        for col in self._static_attributes:
            if col in df.columns and col not in agg_spec:
                agg_spec[col] = (col, "first")

        return df.groupby(key, as_index=False).agg(**agg_spec)

    # --- option-week table --------------------------------------------------

    
    
    def build_option_week_table(self, df: pd.DataFrame) -> pd.DataFrame:
        """Build the one-row-per-option-week feature table (no price targets yet).

        Static attributes are taken from the option-week's horizons, ``Year`` is
        derived from the arrival week, ``AccoTypeRangeCode`` is split into
        ``AccommodationType`` / ``RangeType``, ``Bedrooms`` is parsed to an int,
        and ``is_special_period`` flags any promotional period.

        Args:
            df: Raw weekly extract (market rows still present).

        Returns:
            Feature table keyed by ``(ReservableOptionId, WeekStartDate)``.
        """
        c = self.cols
        collapsed = self.collapse_market_rows(df)

        key = [c.option_id, c.arrival_week]
        agg_spec: dict[str, tuple[str, str]] = {
            c.capacity: (c.capacity, "first"),
        }
        for col in (c.capacity_last_year, c.season, c.acco_type_range, c.special_period):
            if col in collapsed.columns:
                agg_spec[col] = (col, "first")
        for col in self._static_attributes:
            if col in collapsed.columns and col not in agg_spec:
                agg_spec[col] = (col, "first")

        table = collapsed.groupby(key, as_index=False).agg(**agg_spec)
        table = self._add_derived_columns(table)
        return table

    def _add_derived_columns(self, table: pd.DataFrame) -> pd.DataFrame:
        """Add calendar, ordinal-range, capacity-size and flag features.

        Adds ``Year``, ``IsoWeek`` and (if absent) ``ArrivalMonth`` from the
        arrival week; splits ``AccoTypeRangeCode`` into ``AccommodationType`` /
        ``RangeType`` and encodes the latter as the ordinal ``RangeOrdinal``;
        parses ``Bedrooms`` and the messy ``Sleeps`` string into numerics; and
        flags promotional periods.
        """
        c = self.cols
        week = pd.to_datetime(table[c.arrival_week])
        table["Year"] = week.dt.year
        # ISO week sharpens the coarse SeasonalCluster: price varies week-to-week
        # inside a season (e.g. peak vs. shoulder of the summer block).
        table["IsoWeek"] = week.dt.isocalendar().week.astype(int)
        if c.arrival_month not in table.columns:
            table[c.arrival_month] = week.dt.month

        if c.acco_type_range in table.columns:
            split = table[c.acco_type_range].str.split(r"\s*\|\s*", n=1, expand=True)
            table["AccommodationType"] = split[0]
            table["RangeType"] = split[1]

        # The real ordered quality ladder lives in ``AccommodationRange``
        # (Budget…Ultimate); the split ``RangeType`` is anonymised, so the
        # ordinal is built from ``AccommodationRange``. Unknown tiers → NaN.
        if "AccommodationRange" in table.columns:
            table["RangeOrdinal"] = table["AccommodationRange"].map(RANGE_ORDER)

        if "Bedrooms" in table.columns:
            table["Bedrooms"] = (
                table["Bedrooms"].astype(str).str.extract(r"(\d+)").astype(float)
            )

        if "Sleeps" in table.columns:
            table["SleepsNum"] = table["Sleeps"].apply(self._parse_sleeps)

        if c.special_period in table.columns:
            table["is_special_period"] = table[c.special_period].notna().astype(int)
        else:
            table["is_special_period"] = 0
        return table

    @staticmethod
    def _parse_sleeps(value: object) -> float:
        """Parse a messy capacity string to a numeric mid-point.

        Handles plain ``"5"``, ranges ``"4-6"`` / ``"4/6"`` (→ mid-point ``5``),
        and missing values. All embedded integers are averaged, so any future
        ``"a/b"``/``"a-b"`` variant degrades gracefully.

        Args:
            value: Raw ``Sleeps`` cell (string, number or NaN).

        Returns:
            The mean of the integers found, or ``NaN`` if none are present.
        """
        if pd.isna(value):
            return np.nan
        digits = pd.Series(str(value)).str.findall(r"\d+").iloc[0]
        if not digits:
            return np.nan
        return float(np.mean([int(d) for d in digits]))

    # --- initial price ------------------------------------------------------
    def add_initial_prices(self, df: pd.DataFrame, table: pd.DataFrame) -> pd.DataFrame:
        """Attach ``InitialPrice`` and ``InitialPriceLastYear`` to the table.

        Each is the price at the earliest priced horizon (highest
        ``WeekBeforeArrival`` with a non-null price) of its own series, resolved
        independently per option-week.

        Args:
            df: Raw weekly extract (market rows still present).
            table: Output of :meth:`build_option_week_table`.

        Returns:
            ``table`` with the two initial-price columns merged on.
        """
        c = self.cols
        collapsed = self.collapse_market_rows(df)

        initial = self._earliest_priced(collapsed, c.price, "InitialPrice")
        initial_last = self._earliest_priced(
            collapsed, c.price_last_year, "InitialPriceLastYear"
        )

        key = [c.option_id, c.arrival_week]
        out = table.merge(initial, on=key, how="left")
        out = out.merge(initial_last, on=key, how="left")
        return out

    def _earliest_priced(
        self, collapsed: pd.DataFrame, price_col: str, out_name: str
    ) -> pd.DataFrame:
        """Price at the highest ``wba`` with a non-null ``price_col`` per option-week."""
        c = self.cols
        key = [c.option_id, c.arrival_week]
        priced = collapsed.dropna(subset=[price_col])
        if priced.empty:
            empty = collapsed[key].drop_duplicates().copy()
            empty[out_name] = np.nan
            return empty

        earliest = (
            priced.sort_values(c.wba, ascending=False)
            .groupby(key, as_index=False)
            .first()[key + [price_col]]
            .rename(columns={price_col: out_name})
        )
        return earliest

    # --- last-year demand features -----------------------------------------
    def add_demand_features(
        self,
        df: pd.DataFrame,
        table: pd.DataFrame,
        lead_by_season: Optional[dict[str, int]] = None,
    ) -> pd.DataFrame:
        """Attach leakage-safe demand features from the *prior-year* booking curve.

        The opening price is set before any current-season bookings exist, so the
        only demand signal that is safe to use is how the option sold **last**
        year. Two features summarise the prior-season curve per option-week:

        - ``LastYearMaxFill`` — the highest occupancy the option reached last year
          (demand *level*: did it sell out or sit empty?).
        - ``LastYearEarlyFill`` — prior-year occupancy **at the season's lift-off
          lead time** (early booking *pace*): the occupancy at the smallest
          ``WeekBeforeArrival`` still ``>=`` the season cutoff.

        Both are ``NaN`` for options without usable prior-year capacity/bookings
        (the same coverage limit as ``InitialPriceLastYear``); the model pipeline
        imputes them and flags the gap with ``has_last_year``.

        Args:
            df: Raw weekly extract (market rows still present).
            table: Output of :meth:`build_option_week_table` (carries the season).
            lead_by_season: Season → lift-off ``WeekBeforeArrival`` cutoff for the
                ``LastYearEarlyFill`` read. Defaults to :data:`LEAD_BY_SEASON`.

        Returns:
            ``table`` with the two demand columns merged on.
        """
        c = self.cols
        lead = lead_by_season or LEAD_BY_SEASON
        key = [c.option_id, c.arrival_week]
        collapsed = self.collapse_market_rows(df)

        if (
            c.cum_booked_last_year not in collapsed.columns
            or c.capacity_last_year not in collapsed.columns
        ):
            out = table.copy()
            out["LastYearMaxFill"] = np.nan
            out["LastYearEarlyFill"] = np.nan
            return out

        work = collapsed[key + [c.wba, c.cum_booked_last_year, c.capacity_last_year]].copy()
        capacity = work[c.capacity_last_year].where(work[c.capacity_last_year] > 0)
        work["occupancy"] = (work[c.cum_booked_last_year] / capacity).clip(0.0, 1.0)
        work = work.dropna(subset=["occupancy"])

        if work.empty:
            out = table.copy()
            out["LastYearMaxFill"] = np.nan
            out["LastYearEarlyFill"] = np.nan
            return out

        max_fill = (
            work.groupby(key, as_index=False)["occupancy"]
            .max()
            .rename(columns={"occupancy": "LastYearMaxFill"})
        )

        # Early pace = occupancy at the season's lift-off lead time: the smallest
        # WBA still >= the season cutoff (the price/demand read when demand begins).
        if c.season in table.columns:
            work = work.merge(table[key + [c.season]], on=key, how="left")
            work["cutoff"] = work[c.season].map(lead)
        else:
            work["cutoff"] = next(iter(lead.values()), 20)
        eligible = work[work[c.wba] >= work["cutoff"]]
        eligible = eligible if not eligible.empty else work
        early_fill = (
            eligible.sort_values(c.wba, ascending=True)
            .groupby(key, as_index=False)
            .first()[key + ["occupancy"]]
            .rename(columns={"occupancy": "LastYearEarlyFill"})
        )

        out = table.merge(max_fill, on=key, how="left")
        out = out.merge(early_fill, on=key, how="left")
        return out

    # --- lead-time price ----------------------------------------------------
    def add_lead_time_prices(
        self,
        df: pd.DataFrame,
        table: pd.DataFrame,
        lead_by_season: Optional[dict[str, int]] = None,
    ) -> pd.DataFrame:
        """Attach ``LeadTimePrice`` and ``LeadTimePriceLastYear``.

        The lead-time price is the price *standing* when the arrival week reaches its
        season's demand lift-off lead time: the price at the smallest
        ``WeekBeforeArrival`` that is still ``>=`` the season cutoff among priced
        horizons. Unlike the X%-occupancy ``TargetPrice`` it is defined for (almost)
        every priced option, not only those that reach a fill threshold — sidestepping
        the right-censoring of the occupancy target.

        Args:
            df: Raw weekly extract (market rows still present).
            table: Output of :meth:`build_option_week_table` (carries the season).
            lead_by_season: Season → cutoff ``WeekBeforeArrival``. Defaults to
                :data:`LEAD_BY_SEASON`.

        Returns:
            ``table`` with the two lead-time-price columns merged on.
        """
        c = self.cols
        lead = lead_by_season or LEAD_BY_SEASON
        key = [c.option_id, c.arrival_week]
        collapsed = self.collapse_market_rows(df)
        season = table[key + [c.season]] if c.season in table.columns else None

        lead_now = self._price_at_lead(collapsed, c.price, "LeadTimePrice", lead, season)
        lead_last = self._price_at_lead(
            collapsed, c.price_last_year, "LeadTimePriceLastYear", lead, season
        )
        out = table.merge(lead_now, on=key, how="left")
        out = out.merge(lead_last, on=key, how="left")
        return out

    def _price_at_lead(
        self,
        collapsed: pd.DataFrame,
        price_col: str,
        out_name: str,
        lead_by_season: dict[str, int],
        season: Optional[pd.DataFrame],
    ) -> pd.DataFrame:
        """Price at the smallest ``wba`` >= the season's lead cutoff per option-week."""
        c = self.cols
        key = [c.option_id, c.arrival_week]
        if price_col not in collapsed.columns or season is None:
            empty = collapsed[key].drop_duplicates().copy()
            empty[out_name] = np.nan
            return empty

        work = collapsed[key + [c.wba, price_col]].dropna(subset=[price_col])
        work = work.merge(season, on=key, how="left")
        work["cutoff"] = work[c.season].map(lead_by_season)
        eligible = work[work[c.wba] >= work["cutoff"]]
        if eligible.empty:
            empty = collapsed[key].drop_duplicates().copy()
            empty[out_name] = np.nan
            return empty
        # smallest wba >= cutoff == the price active as the week reaches the lead time
        picked = (
            eligible.sort_values(c.wba, ascending=True)
            .groupby(key, as_index=False)
            .first()[key + [price_col]]
            .rename(columns={price_col: out_name})
        )
        return picked

    # --- last-year price dynamics -------------------------------------------
    def add_last_year_dynamics(self, df: pd.DataFrame, table: pd.DataFrame) -> pd.DataFrame:
        """Attach leakage-safe repricing-dynamics features from the prior-year curve.

        From the last-year price series per option-week (earliest → latest):

        - ``LastYearCuts`` / ``LastYearRaises`` — count of strict price decreases /
          increases (how actively the option was repriced last year).
        - ``LastYearDiscountDepth`` — ``(max - min) / max`` of the series (its total
          repricing range), 0 when flat.

        ``NaN`` where the option has no usable prior-year price. Motivated by the EDA
        finding that prices often raise early then cut toward arrival.

        Args:
            df: Raw weekly extract (market rows still present).
            table: Table produced earlier in the pipeline.

        Returns:
            ``table`` with the three dynamics columns merged on.
        """
        c = self.cols
        key = [c.option_id, c.arrival_week]
        out_cols = ["LastYearCuts", "LastYearRaises", "LastYearDiscountDepth"]
        collapsed = self.collapse_market_rows(df)

        if c.price_last_year not in collapsed.columns:
            out = table.copy()
            for col in out_cols:
                out[col] = np.nan
            return out

        work = (
            collapsed[key + [c.wba, c.price_last_year]]
            .dropna(subset=[c.price_last_year])
            .sort_values(c.wba, ascending=False)  # earliest -> latest within each group
        )

        records = []
        for option_key, sub in work.groupby(key, sort=False):
            prices = sub[c.price_last_year].to_numpy()
            diffs = np.diff(prices)
            hi = prices.max()
            depth = float((hi - prices.min()) / hi) if hi > 0 else 0.0
            keys = option_key if isinstance(option_key, tuple) else (option_key,)
            records.append((*keys, int((diffs < -1e-9).sum()),
                            int((diffs > 1e-9).sum()), depth))

        dyn = pd.DataFrame(records, columns=key + out_cols)
        return table.merge(dyn, on=key, how="left")

    # --- realized outcome ---------------------------------------------------
    def add_final_fill(self, df: pd.DataFrame, table: pd.DataFrame) -> pd.DataFrame:
        """Attach ``FinalFill`` — the realized occupancy at arrival (current season).

        ``FinalFill`` = ``CumulativeHistoricalBookedNights / Capacity`` at the latest
        horizon (smallest ``WeekBeforeArrival``, i.e. arrival), clipped to ``[0, 1]``.
        This is a *realized outcome*, used only to label option-weeks as
        over-/on-/under-priced — **never** as a model feature (that would leak the
        future into the price prediction).

        Args:
            df: Raw weekly extract (market rows still present).
            table: Table produced earlier in the pipeline.

        Returns:
            ``table`` with the ``FinalFill`` column merged on.
        """
        c = self.cols
        key = [c.option_id, c.arrival_week]
        collapsed = self.collapse_market_rows(df)

        work = collapsed[key + [c.wba, c.cum_booked, c.capacity]].copy()
        capacity = work[c.capacity].where(work[c.capacity] > 0)
        work["occupancy"] = (work[c.cum_booked] / capacity).clip(0.0, 1.0)

        final = (
            work.sort_values(c.wba, ascending=True)  # arrival (smallest WBA) first
            .groupby(key, as_index=False)
            .first()[key + ["occupancy"]]
            .rename(columns={"occupancy": "FinalFill"})
        )
        return table.merge(final, on=key, how="left")

    # --- target price -------------------------------------------------------
    def add_target_prices(
        self,
        df: pd.DataFrame,
        table: pd.DataFrame,
        season_occupancy: dict[str, float],
    ) -> pd.DataFrame:
        """Attach ``TargetPrice`` and ``TargetPriceLastYear`` at the X%% horizon.

        For each option-week, horizons are scanned from earliest to latest and
        the price is taken at the first horizon where cumulative occupancy meets
        the season's threshold. Option-weeks whose season is missing from
        ``season_occupancy`` keep ``NaN``.

        Args:
            df: Raw weekly extract (market rows still present).
            table: Table produced earlier in the pipeline.
            season_occupancy: Maps a ``SeasonalCluster`` value to its occupancy
                threshold (e.g. ``{"HS": 0.75, "SS": 0.60}``).

        Returns:
            ``table`` with the two target-price columns merged on.
        """
        c = self.cols
        collapsed = self.collapse_market_rows(df)

        target = self._price_at_occupancy(
            collapsed,
            price_col=c.price,
            cum_col=c.cum_booked,
            capacity_col=c.capacity,
            season_occupancy=season_occupancy,
            out_name="TargetPrice",
        )
        target_last = self._price_at_occupancy(
            collapsed,
            price_col=c.price_last_year,
            cum_col=c.cum_booked_last_year,
            capacity_col=c.capacity_last_year
            if c.capacity_last_year in collapsed.columns
            else c.capacity,
            season_occupancy=season_occupancy,
            out_name="TargetPriceLastYear",
            capacity_fallback=c.capacity,
        )

        key = [c.option_id, c.arrival_week]
        out = table.merge(target, on=key, how="left")
        out = out.merge(target_last, on=key, how="left")
        return out

    def _price_at_occupancy(
        self,
        collapsed: pd.DataFrame,
        price_col: str,
        cum_col: str,
        capacity_col: str,
        season_occupancy: dict[str, float],
        out_name: str,
        capacity_fallback: Optional[str] = None,
    ) -> pd.DataFrame:
        """First-horizon price where occupancy meets the season threshold."""
        c = self.cols
        key = [c.option_id, c.arrival_week]

        work = collapsed[
            key + [c.wba, c.season, price_col, cum_col, capacity_col]
        ].copy()

        capacity = work[capacity_col]
        if capacity_fallback is not None:
            capacity = capacity.fillna(collapsed[capacity_fallback])
        capacity = capacity.where(capacity > 0)
        work["occupancy"] = (work[cum_col] / capacity).clip(upper=1.0)
        work["threshold"] = work[c.season].map(season_occupancy)

        reached = work[
            work[price_col].notna()
            & work["threshold"].notna()
            & (work["occupancy"] >= work["threshold"])
        ]
        if reached.empty:
            empty = collapsed[key].drop_duplicates().copy()
            empty[out_name] = np.nan
            return empty

        first = (
            reached.sort_values(c.wba, ascending=False)
            .groupby(key, as_index=False)
            .first()[key + [price_col]]
            .rename(columns={price_col: out_name})
        )
        return first
