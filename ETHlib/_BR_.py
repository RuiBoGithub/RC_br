import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import pickle
# in _BR_.py

from dataclasses import dataclass
from pathlib import Path
import sys
import json
import pandas as pd

from radiation import Location, Window


@dataclass
class RCCase:
    year: int
    loc_json: Path
    geo_json: Path
    default_json: Path
    epw_path: Path
    occupancy_profile_csv: Path
    Zone: object
    supply_system: object
    emission_system: object

    def __post_init__(self):
        cityloc = json.loads(Path(self.loc_json).read_text())

        self.latitude_deg = cityloc["latitude_deg"]
        self.longitude_deg = cityloc["longitude_deg"]

        self.location = Location(epwfile_path=self.epw_path)
        self.geometry = json.loads(Path(self.geo_json).read_text())
        self.default_params = json.loads(Path(self.default_json).read_text())
        self.occupancy_profile = pd.read_csv(self.occupancy_profile_csv)
        
def occupancy_based_ach(
    hour,
    occupancy,
    n_people,
    ach_vent_baseline,
    occupied_ach=None,
    unoccupied_ach=None,
    occupancy_threshold=None,
    unoccupied_ach_fraction=0.1,
    occupancy_threshold_fraction=0.1,
):
    """
    Occupancy-based ventilation controller.

    Default behaviour:
    - occupied_ach = ach_vent_baseline
    - unoccupied_ach = 10% of ach_vent_baseline
    - occupancy_threshold = 10% of n_people

    Optional controller arguments can overwrite these defaults.
    """

    if occupied_ach is None:
        occupied_ach = ach_vent_baseline

    if unoccupied_ach is None:
        unoccupied_ach = unoccupied_ach_fraction * ach_vent_baseline

    if occupancy_threshold is None:
        occupancy_threshold = occupancy_threshold_fraction * n_people

    return occupied_ach if occupancy > occupancy_threshold else unoccupied_ach
    
def calc_ach(n_people,
             fresh_air_lps,
             atrium_ach,
             atrium_volume,
             infl_rate_m3ph_m2,
             geometry):
    
    mech_m3s = n_people * fresh_air_lps / 1000.0
    nat_vent_m3s = atrium_ach * atrium_volume / 3600.0
    vent_m3s = mech_m3s + nat_vent_m3s
    ach_vent = 3600.0 * vent_m3s / geometry["VOLUME"]
    infl_m3ph = infl_rate_m3ph_m2 * geometry["WALL_AREA"]
    ach_infl = infl_m3ph / geometry["VOLUME"]

    return ach_vent, ach_infl

def make_ach(p, geometry, calc_ach):
    return calc_ach(
        n_people=p["max_occupancy"],
        fresh_air_lps=p["fresh_air_lps"],
        atrium_ach=p["atrium_ach"],
        atrium_volume=geometry["ATRIUM_VOLUME"],
        infl_rate_m3ph_m2=p["infl_rate_m3ph_m2"],
        geometry=geometry,
    )

def make_heating_schedule(year, p):
    heating_index = pd.date_range(
        start=f"{year}-01-01 00:00",
        end=f"{year}-12-31 23:00",
        freq="h",
    )

    schedule_mode = p.get("heating_schedule_mode", "original")

    if schedule_mode == "original":
        weekday_profile = (
            [p["t_setback_heating"]] * 8
            + [p["t_set_heating"]] * 11
            + [p["t_setback_heating"]] * 5
        )

        weekend_profile = (
            [p["t_setback_heating"]] * 9
            + [p["t_weekend_heating"]] * 9
            + [p["t_setback_heating"]] * 6
        )

    elif schedule_mode == "8_8_8":
        weekday_profile = (
            [p["t_setback_heating"]] * 8
            + [p["t_set_heating"]] * 8
            + [p["t_setback_heating"]] * 8
        )

        weekend_profile = (
            [p["t_setback_heating"]] * 24
        )

    else:
        raise ValueError(
            f"Unknown heating_schedule_mode: {schedule_mode}. "
            "Use 'original' or '8_8_8'."
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
        total_internal_area=geometry["FLOOR_AREA"] * p["_alpha"],
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

def merge_params(sampled_params, defaults):
    """
    Sampled parameters override deterministic default parameters.
    """
    p = defaults.copy()
    p.update(sampled_params)
    return p

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



# in _BR_.py
def run_model_case(
    case,
    sampled_params=None,
    controller_mode="original",
    occupancy_controller_params=None,
):
    if sampled_params is None:
        sampled_params = {}

    return run_model(
        sampled_params=sampled_params,
        default_params=case.default_params,
        geometry=case.geometry,
        occupancy_profile=case.occupancy_profile,
        location=case.location,
        latitude_deg=case.latitude_deg,
        longitude_deg=case.longitude_deg,
        year=case.year,
        controller_mode=controller_mode,
        occupancy_controller_params=occupancy_controller_params,
        Zone=case.Zone,
        supply_system=case.supply_system,
        emission_system=case.emission_system,
    )

def run_model(
    sampled_params,
    default_params,
    geometry,
    occupancy_profile,
    location,
    latitude_deg,
    longitude_deg,
    year=2023,
    controller_mode="original",
    occupancy_controller_params=None,
    Zone=None,
    supply_system=None,
    emission_system=None,
):
    p = merge_params(sampled_params, default_params)

    HeatingDemand, HeatingEnergy, CoolingDemand, CoolingEnergy = [], [], [], []
    ElectricityOut, IndoorAir, OutsideTemp, SolarGains, COP = [], [], [], [], []
    ach_vent_hourly, ach_infl_hourly, h_ve_adj_hourly = [], [], []

    t_m_prev = 20.0

    heating_schedule = make_heating_schedule(year=year, p=p)

    ach_vent_baseline, ach_infl_baseline = make_ach(
        p=p,
        geometry=geometry,
        calc_ach=calc_ach,
    )

    base_occupancy_controller_params = {
        "n_people": p["max_occupancy"],
        "ach_vent_baseline": ach_vent_baseline,
    }

    if occupancy_controller_params is not None:
        base_occupancy_controller_params.update(occupancy_controller_params)

    occupancy_controller_params = base_occupancy_controller_params

    Office = make_zone(
        p=p,
        geometry=geometry,
        ach_vent=ach_vent_baseline,
        ach_infl=ach_infl_baseline,
        Zone=Zone,
        supply_system=supply_system,
        emission_system=emission_system,
    )

    SouthWindow = Window(
        azimuth_tilt=0,
        alititude_tilt=90,
        glass_solar_transmittance=0.3,
        glass_light_transmittance=0.3,
        area=geometry["WINDOW_AREA"] * p["_beta"],
    )

    for hour in range(8760):
        occupancy = occupancy_profile.loc[hour, "People"] * p["max_occupancy"]

        if controller_mode == "original":
            desired_ach = ach_vent_baseline

        elif controller_mode == "occupancy":
            desired_ach = occupancy_based_ach(
                hour=hour,
                occupancy=occupancy,
                **occupancy_controller_params,
            )

        else:
            raise ValueError(
                f"Unknown controller_mode: {controller_mode}. "
                "Use 'original' or 'occupancy'."
            )

        Office.ach_vent = desired_ach

        ach_vent_hourly.append(Office.ach_vent)
        ach_infl_hourly.append(Office.ach_infl)
        h_ve_adj_hourly.append(Office.h_ve_adj)

        Office.t_set_heating = heating_schedule[hour]

        internal_gains = (
            occupancy * p["gain_per_person"]
            + p["appliance_gains"] * Office.floor_area
        )

        t_out = location.weather_data["drybulb_C"][hour]

        altitude, azimuth = location.calc_sun_position(
            latitude_deg=latitude_deg,
            longitude_deg=longitude_deg,
            year=year,
            hoy=hour,
        )

        SouthWindow.calc_solar_gains(
            sun_altitude=altitude,
            sun_azimuth=azimuth,
            normal_direct_radiation=location.weather_data["dirnorrad_Whm2"][hour],
            horizontal_diffuse_radiation=location.weather_data["difhorrad_Whm2"][hour],
        )

        SouthWindow.calc_illuminance(
            sun_altitude=altitude,
            sun_azimuth=azimuth,
            normal_direct_illuminance=location.weather_data["dirnorillum_lux"][hour],
            horizontal_diffuse_illuminance=location.weather_data["difhorillum_lux"][hour],
        )

        Office.solve_energy(
            internal_gains=internal_gains,
            solar_gains=SouthWindow.solar_gains,
            t_out=t_out,
            t_m_prev=t_m_prev,
        )

        Office.solve_lighting(
            illuminance=SouthWindow.transmitted_illuminance,
            occupancy=occupancy,
        )

        t_m_prev = Office.t_m_next

        fa = geometry["FLOOR_AREA"]

        HeatingDemand.append(Office.heating_demand / 1000.0 / fa)
        HeatingEnergy.append(Office.heating_energy / 1000.0 / fa)
        CoolingDemand.append(Office.cooling_demand / 1000.0 / fa)
        CoolingEnergy.append(Office.cooling_energy / 1000.0 / fa)
        ElectricityOut.append(Office.electricity_out / 1000.0 / fa)
        IndoorAir.append(Office.t_air)
        OutsideTemp.append(t_out)
        SolarGains.append(SouthWindow.solar_gains)
        COP.append(Office.cop)

    annualResults = pd.DataFrame(
        {
            "HeatingDemand": HeatingDemand,
            "HeatingEnergy": HeatingEnergy,
            "CoolingDemand": CoolingDemand,
            "CoolingEnergy": CoolingEnergy,
            "ElectricityOut": ElectricityOut,
            "IndoorAir": IndoorAir,
            "OutsideTemp": OutsideTemp,
            "SolarGains": SolarGains,
            "COP": COP,
            "ach_vent": ach_vent_hourly,
            "ach_infl": ach_infl_hourly,
            "h_ve_adj": h_ve_adj_hourly,
        },
        index=pd.date_range(f"{year}-01-01", periods=8760, freq="h"),
    )

    annual_EUI = annualResults[["HeatingEnergy", "CoolingEnergy"]].sum()

    return annualResults, annual_EUI, Office

import pickle

def run_uncertainty_with_cache(
    config_json,
    make_lhs_samples,
    run_model,
    cache_dir="cache",
    use_cache=False,
):
    config_json = Path(config_json)

    cache_dir = Path(cache_dir)
    cache_path = cache_dir / f"{config_json.stem}_uncertainty_outputs.pkl"
    json_mtime = config_json.stat().st_mtime

    if use_cache:
        cache_dir.mkdir(exist_ok=True)
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
    else:
        print("Cache disabled. Running uncertainty propagation...")

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

    if use_cache:
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

def diagnose_ventilation_glitch(df, title="Ventilation Glitch Diagnostics",
                                start_date=None, end_date=None):
    """
    Create a comprehensive plot to check whether the ventilation controller
    is producing a logic glitch (EUI ~ 0).

    Parameters
    ----------
    df : pd.DataFrame
        Merged DataFrame from annualResults and DebugZone debug log.
        Must contain columns:
        'OutsideTemp', 'IndoorAir', 'has_heating_demand', 'has_cooling_demand',
        't_air_free', 't_air_test', 'delta_t_air', 'energy_demand_unrestricted',
        'energy_demand', 'HeatingDemand', 'CoolingDemand', 'ach_vent', 'h_ve_adj',
        and setpoints 'heating_setpoint', 'cooling_setpoint'.
    title : str
        Overall title for the figure.
    start_date, end_date : str or None
        Slice the DataFrame (e.g., '2023-01-01', '2023-01-14').
        If None, the whole year is plotted.
    """
    # Slice time if requested
    if start_date is not None:
        df = df.loc[start_date:end_date]

    # Create figure with 4 subplots (vertical stack)
    fig, axes = plt.subplots(4, 1, figsize=(14, 14), sharex=True)
    fig.suptitle(title, fontsize=16)

    # ---- Panel 1: Temperatures & setpoints ----
    ax = axes[0]
    ax.plot(df.index, df['OutsideTemp'], label='Outdoor', color='tab:blue',
            linewidth=0.8, alpha=0.7)
    ax.plot(df.index, df['IndoorAir'], label='Indoor Air (actual)', color='tab:red',
            linewidth=0.8)
    ax.plot(df.index, df['t_air_free'], label='Free‑floating Tair', color='tab:orange',
            linestyle='--', linewidth=0.8)
    if 'heating_setpoint' in df.columns:
        ax.plot(df.index, df['heating_setpoint'], label='Heat Setpoint',
                color='black', linestyle=':', linewidth=0.8)
    if 'cooling_setpoint' in df.columns:
        ax.plot(df.index, df['cooling_setpoint'], label='Cool Setpoint',
                color='black', linestyle='-.', linewidth=0.8)
    ax.set_ylabel('Temperature [°C]')
    ax.legend(loc='upper right', ncol=2)
    ax.grid(True, alpha=0.3)

    # ---- Panel 2: Demand flags (binary) ----
    ax = axes[1]
    # Fill areas where heating/cooling demand is True
    heat_flag = df['has_heating_demand'].astype(int) if 'has_heating_demand' in df.columns else None
    cool_flag = df['has_cooling_demand'].astype(int) if 'has_cooling_demand' in df.columns else None
    if heat_flag is not None:
        ax.fill_between(df.index, 0, heat_flag, step='post', alpha=0.5,
                        label='Heating Demand', color='red')
    if cool_flag is not None:
        ax.fill_between(df.index, 0, -cool_flag, step='post', alpha=0.5,
                        label='Cooling Demand', color='blue')
    ax.set_ylim(-1.1, 1.1)
    ax.set_ylabel('Demand Flag')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    # ---- Panel 3: Demand & energy (if available) ----
    ax = axes[2]
    # Plot energy_demand_unrestricted and energy_demand
    if 'energy_demand_unrestricted' in df.columns:
        ax.plot(df.index, df['energy_demand_unrestricted'], label='Unrestricted Demand',
                color='purple', linewidth=0.8, alpha=0.7)
    if 'energy_demand' in df.columns:
        ax.plot(df.index, df['energy_demand'], label='Actual Demand', color='green',
                linewidth=0.8)
    ax.set_ylabel('Power [W]')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    # If all unrestricted demands are NaN or zero, that's suspicious
    if df['energy_demand_unrestricted'].isna().all():
        ax.text(0.5, 0.5, 'UNRESTRICTED DEMAND ALL NaN!', transform=ax.transAxes,
                ha='center', va='center', fontsize=14, color='red')

    # ---- Panel 4: Ventilation & diagnostic slope ----
    ax = axes[3]
    ax2 = ax.twinx()
    ax.plot(df.index, df['ach_vent'], label='ACH vent', color='tab:green', linewidth=0.8)
    ax.set_ylabel('ACH vent', color='tab:green')
    ax.tick_params(axis='y', labelcolor='tab:green')

    # Plot delta_t_air (slope)
    if 'delta_t_air' in df.columns:
        ax2.plot(df.index, df['delta_t_air'], label='delta T (test - free)',
                 color='tab:brown', linestyle='--', linewidth=0.8)
        ax2.set_ylabel('delta T [K]', color='tab:brown')
        ax2.tick_params(axis='y', labelcolor='tab:brown')
        # Highlight near‑zero slope
        ax2.axhline(y=0.1, color='gray', linestyle=':', alpha=0.5)
        ax2.axhline(y=-0.1, color='gray', linestyle=':', alpha=0.5)

    # Combine legends
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, loc='upper right')
    ax.grid(True, alpha=0.3)

    # Format x-axis dates
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    fig.autofmt_xdate()

    plt.tight_layout()
    return fig, axes