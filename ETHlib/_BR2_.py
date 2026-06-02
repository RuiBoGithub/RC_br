import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json

def make_operation_schedule(index, p):
    rows = []

    for ts in index:
        is_weekend = ts.weekday() >= 5
        is_daytime = 8 <= ts.hour < 18

        if is_weekend:
            n_people = p.get("n_people_weekend_day", 0) if is_daytime else 0
            atrium_ach = (
                p.get("atrium_ach_weekend_day", 0.0)
                if is_daytime
                else p.get("atrium_ach_night", 0.0)
            )
        else:
            n_people = p.get("n_people_weekday_day", 865) if is_daytime else 0
            atrium_ach = (
                p.get("atrium_ach_weekday_day", 2.0)
                if is_daytime
                else p.get("atrium_ach_night", 0.0)
            )

        rows.append({
            "Timestamp": ts,
            "n_people": n_people,
            "atrium_ach": atrium_ach,
        })

    return pd.DataFrame(rows).set_index("Timestamp")

def make_ach_schedule(index, p, geometry, calc_ach):
    op = make_operation_schedule(index, p)

    rows = []

    for ts, row in op.iterrows():
        ach_vent, ach_infl = calc_ach(
            n_people=row["n_people"],
            fresh_air_lps=p.get("fresh_air_lps", 12.0),
            atrium_ach=row["atrium_ach"],
            atrium_volume=p.get("atrium_volume", 5739.53),
            infl_rate_m3ph_m2=p["infl_rate_m3ph_m2"],
            geometry=geometry,
        )

        rows.append({
            "Timestamp": ts,
            "n_people": row["n_people"],
            "atrium_ach": row["atrium_ach"],
            "ach_vent": ach_vent,
            "ach_infl": ach_infl,
        })

    return pd.DataFrame(rows).set_index("Timestamp")

def calc_ach(n_people,
             fresh_air_lps,
             atrium_ach,
             atrium_volume,
             infl_rate_m3ph_m2,
             geometry):
    
    # mechanical ventilation based on N_p [m3/s]
    mech_m3s = n_people * fresh_air_lps / 1000.0
    # atrium natural ventilation from ACH [m3/s]
    nat_vent_m3s = atrium_ach * atrium_volume / 3600.0
    vent_m3s = mech_m3s + nat_vent_m3s
    ach_vent = 3600.0 * vent_m3s / geometry["VOLUME"]
    # infiltration from permeability at pressure test condition [m3/h/m2]
    infl_m3ph = infl_rate_m3ph_m2 * geometry["WALL_AREA"]
    ach_infl = infl_m3ph / geometry["VOLUME"]

    return ach_vent, ach_infl

def make_ach(p, geometry, calc_ach):
    return calc_ach(
        n_people=865,
        fresh_air_lps=12.0,
        atrium_ach=2.0,
        atrium_volume=5739.53,
        infl_rate_m3ph_m2=p["infl_rate_m3ph_m2"],
        geometry=geometry,
    )

def make_heating_schedule(year, p):
    heating_index = pd.date_range(
        start=f"{year}-01-01 00:00",
        end=f"{year}-12-31 23:00",
        freq="h"
    )

    weekday_profile = (
        [p["t_setback_heating"]] * 8
        + [p["t_set_heating"]] * 10
        + [p["t_setback_heating"]] * 6
    )

    weekend_profile = (
        [p["t_setback_heating"]] * 8
        + [p["t_weekend_heating"]] * 10
        + [p["t_setback_heating"]] * 6
    )

    heating_schedule = []

    for ts in heating_index:
        if ts.weekday() >= 5:
            heating_schedule.append(weekend_profile[ts.hour])
        else:
            heating_schedule.append(weekday_profile[ts.hour])

    return heating_schedule

def make_zone(
    p,
    geometry,
    ach_vent,
    ach_infl,
    Zone,
    supply_system,
    emission_system,
):
    return Zone(
        window_area=geometry["WINDOW_AREA"],
        walls_area=geometry["WALL_AREA"],
        floor_area=geometry["FLOOR_AREA"],
        room_vol=geometry["VOLUME"],
        total_internal_area=geometry["FLOOR_AREA"] * geometry["_alpha"],

        thermal_capacitance_per_floor_area=p["thermal_capacitance_per_floor_area"],
        u_walls=p["u_walls"],
        u_windows=p["u_windows"],

        ach_vent=ach_vent,
        ach_infl=ach_infl,

        lighting_load=p["lighting_load"],
        t_set_heating=p["t_set_heating"],
        t_set_cooling=p["t_set_cooling"],
        ventilation_efficiency=p["ventilation_efficiency"],

        max_cooling_energy_per_floor_area=-np.inf,
        max_heating_energy_per_floor_area=np.inf,

        heating_supply_system=supply_system.DirectHeater,
        cooling_supply_system=supply_system.DirectCooler,
        heating_emission_system=emission_system.AirConditioning,
        cooling_emission_system=emission_system.AirConditioning,
    )





import numpy as np
def merge_params(sampled_params, defaults):
    """
    Sampled parameters override deterministic default parameters.
    """
    p = defaults.copy()
    p.update(sampled_params)
    return p




def make_hr_eff(p, mech_ach):
    hr_eff = []
    hour_i = 0

    winter_eff = p["ventilation_efficiency"]

    for m_days, eff_winter in [
        (31, winter_eff), (28, winter_eff), (31, winter_eff),
        (30, winter_eff), (31, 0.0), (30, 0.0),
        (31, 0.0), (31, 0.0), (30, 0.0),
        (31, winter_eff), (30, winter_eff), (31, winter_eff),
    ]:
        for _ in range(m_days * 24):
            hr_eff.append(eff_winter if mech_ach[hour_i] > 0 else 0.0)
            hour_i += 1

    return hr_eff

import pickle
from pathlib import Path
import pandas as pd
def summarise_uncertainty_outputs(
    outputs,
    time_col,
    value_col="HeatingEnergy",
):
    return (
        outputs
        .groupby(time_col)[value_col]
        .quantile([0.05, 0.25, 0.50, 0.75, 0.95])
        .unstack()
        .rename(columns={
            0.05: "q05",
            0.25: "q25",
            0.50: "median",
            0.75: "q75",
            0.95: "q95",
        })
    )


def run_uncertainty_with_cache(
    config_json,
    make_lhs_samples,
    run_model,
    cache_dir="cache",
):
    config_json = Path(config_json)

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(exist_ok=True)

    cache_path = cache_dir / f"{config_json.stem}_uncertainty_outputs.pkl"
    json_mtime = config_json.stat().st_mtime

    print(f"Using cache file: {cache_path}")

    if cache_path.exists():
        with open(cache_path, "rb") as f:
            cached = pickle.load(f)

        if cached.get("json_mtime") == json_mtime:
            print(f"Loading cached results from: {cache_path}")

            return (
                cached["samples"],
                cached["hourly_outputs"],
                cached["hourly_sim_summary"],
                cached["daily_outputs"],
                cached["daily_sim_summary"],
            )

    print("JSON changed or no cache found. Running uncertainty propagation...")

    samples = make_lhs_samples(config_json)

    hourly_outputs = []
    daily_outputs = []

    for i, row in samples.iterrows():
        sampled_params = row.to_dict()

        annualResults_i, annual_EUI_i = run_model(sampled_params)

        # Hourly outputs
        tmp_hourly = annualResults_i.reset_index()
        tmp_hourly = tmp_hourly.rename(columns={"index": "DateTime"})
        tmp_hourly["sample_id"] = i
        hourly_outputs.append(tmp_hourly)

        # Daily outputs
        heating_daily_i = (
            annualResults_i["HeatingEnergy"]
            .resample("D")
            .sum()
        )

        tmp_daily = heating_daily_i.rename("HeatingEnergy").reset_index()
        tmp_daily = tmp_daily.rename(columns={"index": "Date"})
        tmp_daily["sample_id"] = i
        daily_outputs.append(tmp_daily)

    hourly_outputs = pd.concat(hourly_outputs, ignore_index=True)
    daily_outputs = pd.concat(daily_outputs, ignore_index=True)

    hourly_sim_summary = summarise_uncertainty_outputs(
        outputs=hourly_outputs,
        time_col="DateTime",
        value_col="HeatingEnergy",
    )

    daily_sim_summary = summarise_uncertainty_outputs(
        outputs=daily_outputs,
        time_col="Date",
        value_col="HeatingEnergy",
    )

    cached = {
        "config_path": str(config_json),
        "json_mtime": json_mtime,
        "samples": samples,
        "hourly_outputs": hourly_outputs,
        "hourly_sim_summary": hourly_sim_summary,
        "daily_outputs": daily_outputs,
        "daily_sim_summary": daily_sim_summary,
    }

    with open(cache_path, "wb") as f:
        pickle.dump(cached, f)

    print(f"Cached results saved to: {cache_path}")

    return (
        samples,
        hourly_outputs,
        hourly_sim_summary,
        daily_outputs,
        daily_sim_summary,
    )


def lhs_uniform(n, d, seed=42):
    rng = np.random.default_rng(seed)
    lhs = np.zeros((n, d))

    for j in range(d):
        cut = np.linspace(0, 1, n + 1)
        u = rng.uniform(cut[:-1], cut[1:])
        rng.shuffle(u)
        lhs[:, j] = u

    return lhs


def make_lhs_samples(config_json):
    config_json = Path(config_json)

    with open(config_json, "r") as f:
        config = json.load(f)

    n = config["N"]
    seed = config.get("seed", 42)
    params = config["parameters"]

    names = list(params.keys())
    lhs_unit = lhs_uniform(n=n, d=len(names), seed=seed)

    samples = pd.DataFrame(index=range(n))

    for j, name in enumerate(names):
        spec = params[name]

        if spec["distribution"] != "uniform":
            raise ValueError(
                f"Unsupported distribution for {name}: {spec['distribution']}"
            )

        lower = spec["lower"]
        upper = spec["upper"]

        samples[name] = lower + lhs_unit[:, j] * (upper - lower)

    return samples

def plot_heating_coverage(
    sim_summary,
    meter_series,
    start_date=None,
    end_date=None,
    meter_col_name="MeteredHeating",
    time_label="Date",
    y_label=None,
    title=None,
):
    sim_view = sim_summary.copy()
    meter_view = meter_series.copy()

    if start_date is not None:
        start_ts = pd.Timestamp(start_date)
        sim_view = sim_view.loc[start_ts:]
        meter_view = meter_view.loc[start_ts:]
    else:
        start_ts = sim_view.index.min()

    if end_date is not None:
        end_ts = pd.Timestamp(end_date)
        sim_view = sim_view.loc[:end_ts]
        meter_view = meter_view.loc[:end_ts]
    else:
        end_ts = sim_view.index.max()

    coverage = sim_view.join(
        meter_view.rename(meter_col_name),
        how="inner"
    )

    coverage["covered_q05_q95"] = coverage[meter_col_name].between(
        coverage["q05"],
        coverage["q95"]
    )

    coverage["covered_IQR"] = coverage[meter_col_name].between(
        coverage["q25"],
        coverage["q75"]
    )

    coverage_q05_q95 = coverage["covered_q05_q95"].mean() * 100
    coverage_iqr = coverage["covered_IQR"].mean() * 100

    print(f"Viewing period: {start_ts} to {end_ts}")
    print(f"Number of matched timesteps: {len(coverage)}")
    print(f"Coverage by q05-q95: {coverage_q05_q95:.1f}%")
    print(f"Coverage by IQR: {coverage_iqr:.1f}%")

    fig, ax = plt.subplots(figsize=(12, 5))

    ax.fill_between(
        coverage.index,
        coverage["q05"],
        coverage["q95"],
        alpha=0.20,
        label=r"Simulated $q_{05}$--$q_{95}$"
    )

    ax.fill_between(
        coverage.index,
        coverage["q25"],
        coverage["q75"],
        alpha=0.35,
        label="Simulated IQR"
    )

    ax.plot(
        coverage.index,
        coverage["median"],
        linewidth=2,
        label="Simulated median"
    )

    ax.plot(
        coverage.index,
        coverage[meter_col_name],
        linestyle="--",
        linewidth=2,
        label="Metered heating"
    )

    ax.set_xlabel(time_label)

    if y_label is None:
        y_label = r"Heating energy intensity (kWh m$^{-2}$)"

    ax.set_ylabel(y_label)

    if title is None:
        title = (
            "Coverage of metered heating by propagated uncertainty\n"
            f"{start_ts} to {end_ts}"
        )

    ax.set_title(title)
    ax.legend(frameon=False)
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    plt.show()

    return coverage

def load_meter_heating(
    meter_path="../_data/Metering_ISO.csv",
    heating_col="Main Heating",
    floor_area=None,
    freq="D",
    dayfirst=False,
):
    meter = pd.read_csv(meter_path)

    meter["Timestamp"] = pd.to_datetime(
        meter["Timestamp"],
        dayfirst=dayfirst,
        errors="coerce"
    )

    meter = (
        meter
        .dropna(subset=["Timestamp"])
        .sort_values("Timestamp")
        .set_index("Timestamp")
    )

    print("Metering Timestamp:", meter.index.min(), "~", meter.index.max())

    meter_heating = meter[heating_col].resample(freq).sum()

    if floor_area is not None:
        meter_heating = meter_heating / floor_area

    meter_heating.name = "MeteredHeating"

    return meter_heating