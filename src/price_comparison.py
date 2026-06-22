"""Compare the initial / target / lead-time Random Forest models on shared options.

The price pipelines are trained on the same option-week table, so their test rows
share a ``ReservableOptionId``. :func:`compare_rf_predictions` joins the models'
Random-Forest test predictions (and the true prices they were trained on) on that
identifier and reports how the X%%-occupancy *target* price and the *lead-time*
price sit relative to the *opening* price for the same accommodation-week.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from src.price_model_pipeline import PriceModelResult


def compare_rf_predictions(
    initial_result: PriceModelResult,
    target_result: PriceModelResult,
    leadtime_result: Optional[PriceModelResult] = None,
    option_id_column: str = "ReservableOptionId",
) -> pd.DataFrame:
    """Join the models' Random-Forest test predictions on ``ReservableOptionId``.

    Args:
        initial_result: Pipeline result with ``InitialPrice`` as the target.
        target_result: Pipeline result with ``TargetPrice`` as the target.
        leadtime_result: Optional pipeline result with ``LeadTimePrice`` as the
            target; when given, its actual/predicted prices are added.
        option_id_column: Shared identifier to join on.

    Returns:
        One row per shared option carrying the actual and RF-predicted initial,
        target (and optionally lead-time) prices, plus the opening→target absolute
        and percentage difference.
    """
    initial = _prediction_frame(
        initial_result, option_id_column, "InitialPrice",
        "actual_init", "pred_init", keep_context=True,
    )
    target = _prediction_frame(
        target_result, option_id_column, "TargetPrice",
        "actual_target", "pred_target", keep_context=False,
    )

    compare = initial.merge(
        target, on=option_id_column, how="inner", validate="one_to_one"
    )
    if leadtime_result is not None:
        lead = _prediction_frame(
            leadtime_result, option_id_column, "LeadTimePrice",
            "actual_lead", "pred_lead", keep_context=False,
        )
        compare = compare.merge(
            lead, on=option_id_column, how="inner", validate="one_to_one"
        )

    compare["diff"] = compare["pred_target"] - compare["pred_init"]
    compare["diff_actual"] = compare["actual_target"] - compare["actual_init"]
    compare["pct_diff"] = np.where(
        compare["pred_init"] != 0,
        compare["diff"] / compare["pred_init"] * 100,
        np.nan,
    )
    return compare


def summarise_comparison(compare: pd.DataFrame) -> None:
    """Print headline statistics for a comparison frame.

    Args:
        compare: Output of :func:`compare_rf_predictions`.
    """
    corr = compare[["pred_init", "pred_target"]].corr().iloc[0, 1]
    print(f"shared options: {len(compare):,}")
    print(
        f"mean actual_init   {compare['actual_init'].mean():8.2f}  | "
        f"mean pred_init   {compare['pred_init'].mean():8.2f}"
    )
    print(
        f"mean actual_target {compare['actual_target'].mean():8.2f}  | "
        f"mean pred_target {compare['pred_target'].mean():8.2f}"
    )
    if "pred_lead" in compare.columns:
        print(
            f"mean actual_lead   {compare['actual_lead'].mean():8.2f}  | "
            f"mean pred_lead   {compare['pred_lead'].mean():8.2f}"
        )
    print(
        f"mean diff (target - init) {compare['diff'].mean():8.2f}  "
        f"(median {compare['diff'].median():.2f})"
    )
    print(
        f"mean pct_diff {compare['pct_diff'].mean():6.1f}%  "
        f"(median {compare['pct_diff'].median():.1f}%)"
    )
    print(f"corr(pred_init, pred_target) {corr:.3f}")


def _prediction_frame(
    result: PriceModelResult,
    option_id_column: str,
    target_column: str,
    actual_name: str,
    pred_name: str,
    keep_context: bool,
) -> pd.DataFrame:
    """Build a one-row-per-option frame of actual and RF-predicted prices.

    Args:
        result: Fitted pipeline result to read test predictions from.
        option_id_column: Identifier to collapse on.
        target_column: Actual price column in ``result.test_meta``.
        actual_name: Output name for the actual price.
        pred_name: Output name for the RF prediction.
        keep_context: Whether to also keep ``WeekStartDate`` / ``SeasonalCluster``
            (kept once, from the initial frame, to avoid ``_x`` / ``_y`` suffixes).
    """
    frame = result.test_meta.copy()
    frame[pred_name] = result.rf_predictions
    frame = frame.rename(columns={target_column: actual_name})

    keep = [option_id_column]
    if keep_context:
        keep += ["WeekStartDate", "SeasonalCluster"]
    keep += [actual_name, pred_name]
    keep = [c for c in keep if c in frame.columns]
    frame = frame[keep]
    return frame.groupby(option_id_column, as_index=False).first()
