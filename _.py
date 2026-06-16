import json
from numbers import Number
from pathlib import Path
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
controller_mode = 'occupancy'
def get_numbered_filename(folder, stem, suffix=".json", date_fmt="%Y%m%d"):

    folder = Path(folder)
    date_str = datetime.now().strftime(date_fmt)

    i = 1
    while True:
        filename = f"{stem}_{date_str}_{i:03d}{suffix}"
        path = folder / filename

        if not path.exists():
            return filename

        i += 1

def make_uq_json_from_defaults(
    default_params,
    exclude_params=None,
    fixed_ranges=None,
    param_variations=None,
    N=10,
    seed=42,
    variation=0.10,
    distribution="uniform",
    save_dir=Path("_json/UC"),
    output_stem="uq_default_pm10",
    round_digits=6,
):

    if exclude_params is None:
        exclude_params = []

    if fixed_ranges is None:
        fixed_ranges = {}

    if param_variations is None:
        param_variations = {}

    exclude_params = set(exclude_params)

    parameters = {}

    for name, value in default_params.items():

        if name in exclude_params:
            continue

        if not isinstance(value, Number):
            continue

        # 1. Use fixed range if explicitly defined
        if name in fixed_ranges:
            lower, upper = fixed_ranges[name]

        # 2. Otherwise use parameter-specific variation if defined
        else:
            this_variation = param_variations.get(name, variation)

            lower = value * (1.0 - this_variation)
            upper = value * (1.0 + this_variation)

        parameters[name] = {
            "distribution": distribution,
            "lower": round(lower, round_digits),
            "upper": round(upper, round_digits),
        }

    config = {
        "N": N,
        "seed": seed,
        "parameters": parameters,
    }

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    filename = get_numbered_filename(
        folder=save_dir,
        stem=output_stem,
        suffix=".json",
    )

    output_path = save_dir / filename

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    print(f"Saved UC JSON to: {output_path}")

    return config, output_path

##

##
from tqdm.auto import tqdm

def show_progress(
    iterable,
    total=None,
    desc="Running",
    unit="item",
    mininterval=1.0,
    miniters=1,
):
    return tqdm(
        iterable,
        total=total,
        desc=desc,
        unit=unit,
        mininterval=mininterval,
        miniters=miniters,
    )


##

def make_8_8_8_suppressed_electricity_profile(
    base_profile,
    year=2023,
    col="People",
    unoccupied_value=0.1,
):
    profile = base_profile.copy().reset_index(drop=True)

    idx = pd.date_range(
        start=f"{year}-01-01 00:00",
        end=f"{year}-12-31 23:00",
        freq="h",
    )

    if len(profile) != len(idx):
        raise ValueError(
            f"Profile length is {len(profile)}, but expected {len(idx)} for {year}."
        )

    is_weekday = idx.weekday < 5
    is_expected_occupied = is_weekday & (idx.hour >= 8) & (idx.hour < 16)

    profile.loc[~is_expected_occupied, col] = unoccupied_value

    return profile


def calc_metrics(sim, obs):
    df = pd.concat(
        [sim.rename("sim"), obs.rename("obs")],
        axis=1,
    ).dropna()

    error = df["obs"] - df["sim"]
    obs_mean = df["obs"].mean()

    cvrmse = np.sqrt(np.mean(error ** 2)) / obs_mean * 100
    nmbe = error.sum() / (len(df) * obs_mean) * 100

    return cvrmse, nmbe


def compare_lhs_to_meter(
    lhs_results,
    meter_series,
    variable="HeatingEnergy",
    freq="D",
    freq_label="daily",
    period_start=None,
    period_end=None,
):
    rows = []

    meter = meter_series.copy()
    meter.index = pd.to_datetime(meter.index)

    if period_start is not None or period_end is not None:
        meter = meter.loc[period_start:period_end]

    if freq is not None:
        meter = meter.resample(freq).sum()

    for sample_id, res in lhs_results.items():

        sim = res[variable].copy()
        sim.index = pd.to_datetime(sim.index)

        if period_start is not None or period_end is not None:
            sim = sim.loc[period_start:period_end]

        if freq is not None:
            sim = sim.resample(freq).sum()

        cvrmse, nmbe = calc_metrics(sim, meter)

        rows.append((sample_id, freq_label, cvrmse, nmbe))

    return pd.DataFrame(
        rows,
        columns=["sample_id", "freq", "cvrmse", "nmbe"],
    ).set_index("sample_id")




def make_posterior_Y_from_results(
    posterior_results,
    variable="HeatingEnergy",
    freq="D",
    start_date="2023-01-20",
    end_date="2023-03-10",
):
    Y = {}

    for sample_id, res in posterior_results.items():
        sim = res[variable].copy()
        sim.index = pd.to_datetime(sim.index)

        sim = (
            sim.loc[start_date:end_date]
               .resample(freq)
               .sum()
        )

        Y[sample_id] = sim

    return pd.DataFrame(Y)

def make_sim_summary(Y):
    """
    Convert simulation ensemble Y into summary quantiles.

    Y:
        rows = timestamps
        columns = simulation samples
    """

    sim_summary = pd.DataFrame(index=Y.index)

    sim_summary["q05"] = Y.quantile(0.05, axis=1)
    sim_summary["q25"] = Y.quantile(0.25, axis=1)
    sim_summary["median"] = Y.median(axis=1)
    sim_summary["q75"] = Y.quantile(0.75, axis=1)
    sim_summary["q95"] = Y.quantile(0.95, axis=1)

    return sim_summary

def compute_ecm_savings(
    all_results,
    baseline_name="Baseline",
    variable="HeatingEnergy",
    period_start=None,
    period_end=None,
    freq=None,
):
    baseline_results = all_results[baseline_name]
    rows = []

    for ecm_name, ecm_results in all_results.items():
        if ecm_name == baseline_name:
            continue

        common_ids = [
            sample_id for sample_id in baseline_results
            if sample_id in ecm_results
        ]

        for sample_id in common_ids:
            baseline = baseline_results[sample_id][variable].copy()
            ecm = ecm_results[sample_id][variable].copy()

            baseline.index = pd.to_datetime(baseline.index)
            ecm.index = pd.to_datetime(ecm.index)

            if period_start is not None or period_end is not None:
                baseline = baseline.loc[period_start:period_end]
                ecm = ecm.loc[period_start:period_end]

            if freq is not None:
                baseline = baseline.resample(freq).sum()
                ecm = ecm.resample(freq).sum()

            df = pd.concat(
                [
                    baseline.rename("baseline"),
                    ecm.rename("ecm_energy"),
                ],
                axis=1,
                join="inner",
            ).dropna()

            baseline_energy = df["baseline"].sum()
            ecm_energy = df["ecm_energy"].sum()
            saving_abs = baseline_energy - ecm_energy

            rows.append(
                {
                    "ecm": ecm_name,
                    "sample_id": sample_id,
                    "period_start": period_start,
                    "period_end": period_end,
                    "freq": freq,
                    "baseline": baseline_energy,
                    "ecm_energy": ecm_energy,
                    "saving_abs": saving_abs,
                    "saving_rel": saving_abs / baseline_energy * 100.0,
                }
            )

    return pd.DataFrame(rows)

def plot_ecm_savings_pair(
    savings_df,
    kpi_label="Heating energy",
    abs_unit="kWh m$^{-2}$",
    figsize=(12, 8),
    rot=30,
):
    plot_specs = {
        "saving_abs": (
            f"Absolute {kpi_label.lower()} saving ({abs_unit})",
            f"Absolute {kpi_label.lower()} savings across posterior samples",
        ),
        "saving_rel": (
            f"Relative {kpi_label.lower()} saving (%)",
            f"Relative {kpi_label.lower()} savings across posterior samples",
        ),
    }

    for value_col, (y_label, title) in plot_specs.items():
        fig, ax = plt.subplots(figsize=figsize)

        savings_df.boxplot(
            column=value_col,
            by="ecm",
            ax=ax,
            rot=rot,
        )

        ax.axhline(0, linestyle="--", linewidth=1)
        ax.set_ylabel(y_label)
        ax.set_xlabel("")
        ax.set_title(title)
        ax.grid(True, axis="y", alpha=0.3)

        plt.suptitle("")
        plt.tight_layout()
        plt.show()