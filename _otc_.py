# _export_outcomes.py

from pathlib import Path
import json
import pandas as pd

from _ import *   # contains make_posterior_Y_from_results, calc_metrics, compute_ecm_savings


def export_posterior_outcomes(
    all_results,
    meter_daily,
    start_date,
    end_date,
    experiment_id,
    engine="python",
    model_name="5R1C",
    controller_mode=None,
    baseline_name="Baseline",
    variable="HeatingEnergy",
    out_dir=Path("_results"),
):
    run_dir = Path(out_dir) / experiment_id
    run_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "experiment_id": experiment_id,
        "engine": engine,
        "model_name": model_name,
        "controller_mode": controller_mode,
        "baseline_name": baseline_name,
        "variable": variable,
        "period_start": start_date,
        "period_end": end_date,
    }

    with open(run_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4, default=str)

    # Baseline posterior predictions
    baseline_hourly_Y = make_posterior_Y_from_results(
        posterior_results=all_results[baseline_name],
        variable=variable,
        freq="H",
        start_date=start_date,
        end_date=end_date,
    )

    baseline_daily_Y = make_posterior_Y_from_results(
        posterior_results=all_results[baseline_name],
        variable=variable,
        freq="D",
        start_date=start_date,
        end_date=end_date,
    )

    baseline_hourly_df = (
        baseline_hourly_Y
        .rename_axis("timestamp")
        .reset_index()
        .melt(
            id_vars="timestamp",
            var_name="sample_id",
            value_name="heating_energy",
        )
    )

    baseline_daily_df = (
        baseline_daily_Y
        .rename_axis("timestamp")
        .reset_index()
        .melt(
            id_vars="timestamp",
            var_name="sample_id",
            value_name="heating_energy",
        )
    )

    for df, freq in [
        (baseline_hourly_df, "hourly"),
        (baseline_daily_df, "daily"),
    ]:
        df.insert(0, "experiment_id", experiment_id)
        df.insert(1, "engine", engine)
        df.insert(2, "scenario", baseline_name)
        df.insert(3, "freq", freq)

    baseline_period_df = (
        baseline_daily_df
        .groupby(
            ["experiment_id", "engine", "scenario", "sample_id"],
            as_index=False,
        )["heating_energy"]
        .sum()
        .rename(columns={"heating_energy": "heating_energy_period"})
    )

    baseline_period_df.insert(3, "freq", "period")

    # Daily CVRMSE/NMBE per posterior sample
    meter_eval = meter_daily.copy()
    meter_eval.index = pd.to_datetime(meter_eval.index)
    meter_eval = meter_eval.loc[start_date:end_date]

    metric_rows = []

    for sample_id in baseline_daily_Y.columns:
        sim_eval = baseline_daily_Y[sample_id].copy()
        sim_eval.index = pd.to_datetime(sim_eval.index)
        sim_eval = sim_eval.loc[start_date:end_date]

        sim_eval, obs_eval = sim_eval.align(meter_eval, join="inner")

        cvrmse, nmbe = calc_metrics(
            sim=sim_eval,
            obs=obs_eval,
        )

        metric_rows.append(
            {
                "experiment_id": experiment_id,
                "engine": engine,
                "scenario": baseline_name,
                "sample_id": sample_id,
                "period_start": start_date,
                "period_end": end_date,
                "freq": "daily",
                "n_points": len(sim_eval),
                "cvrmse": cvrmse,
                "nmbe": nmbe,
            }
        )

    metric_df = pd.DataFrame(metric_rows)

    metric_summary_df = (
        metric_df[["cvrmse", "nmbe"]]
        .agg(["mean", "median", "std", "min", "max"])
        .reset_index()
        .rename(columns={"index": "statistic"})
    )

    # ECM savings per posterior sample
    savings_df = compute_ecm_savings(
        all_results=all_results,
        baseline_name=baseline_name,
        variable=variable,
        period_start=start_date,
        period_end=end_date,
        freq="D",
    )

    savings_df.insert(0, "experiment_id", experiment_id)
    savings_df.insert(1, "engine", engine)

    savings_summary_df = (
        savings_df
        .groupby(["experiment_id", "engine", "ecm"], as_index=False)
        .agg(
            n_samples=("sample_id", "nunique"),
            saving_abs_mean=("saving_abs", "mean"),
            saving_abs_median=("saving_abs", "median"),
            saving_abs_std=("saving_abs", "std"),
            saving_abs_q05=("saving_abs", lambda x: x.quantile(0.05)),
            saving_abs_q95=("saving_abs", lambda x: x.quantile(0.95)),
            saving_rel_mean=("saving_rel", "mean"),
            saving_rel_median=("saving_rel", "median"),
            saving_rel_std=("saving_rel", "std"),
            saving_rel_q05=("saving_rel", lambda x: x.quantile(0.05)),
            saving_rel_q95=("saving_rel", lambda x: x.quantile(0.95)),
        )
    )

    # Save outputs
    tables = {
        "baseline_prediction_hourly": baseline_hourly_df,
        "baseline_prediction_daily": baseline_daily_df,
        "baseline_prediction_period": baseline_period_df,
        "baseline_metrics_daily": metric_df,
        "baseline_metrics_daily_summary": metric_summary_df,
        "ecm_savings_daily": savings_df,
        "ecm_savings_daily_summary": savings_summary_df,
    }

    for name, df in tables.items():
        df.to_csv(run_dir / f"{name}.csv", index=False)

    return tables