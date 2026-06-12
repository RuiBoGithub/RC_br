import json
from numbers import Number
from pathlib import Path
from datetime import datetime
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
##
def run_oat_from_uq_json(
    uq_json_path,
    default_params,
    controller_mode="occupancy",
    save_dir=None,
    period_start="2023-01-20",
    period_end="2023-03-20",
):

    uq_json_path = Path(uq_json_path)
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    with open(uq_json_path, "r", encoding="utf-8") as f:
        uq_config = json.load(f)

    oat_results = {}
    oat_euis = {}

    def calc_period_eui(res):
        res_period = res.loc[period_start:period_end]

        heating = res_period["HeatingEnergy"].sum()
        cooling = res_period["CoolingEnergy"].sum()

        return {
            "HeatingEnergy": heating,
            "CoolingEnergy": cooling,
        }

    summary_rows = []

    n_runs = 1 + 2 * len(uq_config["parameters"])

    with show_progress(
        range(n_runs),
        total=n_runs,
        desc="Running OAT",
        unit="case",
    ) as pbar:

        # Baseline run
        baseline_params = default_params.copy()

        oat_results["baseline"], _, _ = run_model_case(
            case=case,
            sampled_params=baseline_params,
            controller_mode=controller_mode,
        )

        pbar.update(1)

        oat_euis["baseline"] = calc_period_eui(oat_results["baseline"])

        baseline_heating = oat_euis["baseline"]["HeatingEnergy"]
        baseline_cooling = oat_euis["baseline"]["CoolingEnergy"]
        baseline_total = baseline_heating + baseline_cooling

        summary_rows.append({
            "parameter": "baseline",
            "case": "baseline",
            "value": None,
            "heating_eui": baseline_heating,
            "cooling_eui": baseline_cooling,
            "total_hvac_eui": baseline_total,
            "delta_heating_eui": 0.0,
            "delta_cooling_eui": 0.0,
            "delta_total_hvac_eui": 0.0,
        })

        # OAT runs
        for param_name, param_config in uq_config["parameters"].items():

            for bound_name in ["lower", "upper"]:

                sampled_params = default_params.copy()
                sampled_params[param_name] = param_config[bound_name]

                case_name = f"OAT_{param_name}_{bound_name}"

                oat_results[case_name], _, _ = run_model_case(
                    case=case,
                    sampled_params=sampled_params,
                    controller_mode=controller_mode,
                )

                pbar.update(1)

                oat_euis[case_name] = calc_period_eui(oat_results[case_name])

                heating_eui = oat_euis[case_name]["HeatingEnergy"]
                cooling_eui = oat_euis[case_name]["CoolingEnergy"]
                total_hvac_eui = heating_eui + cooling_eui

                summary_rows.append({
                    "parameter": param_name,
                    "case": bound_name,
                    "value": param_config[bound_name],
                    "heating_eui": heating_eui,
                    "cooling_eui": cooling_eui,
                    "total_hvac_eui": total_hvac_eui,
                    "delta_heating_eui": heating_eui - baseline_heating,
                    "delta_cooling_eui": cooling_eui - baseline_cooling,
                    "delta_total_hvac_eui": total_hvac_eui - baseline_total,
                })

    oat_summary = pd.DataFrame(summary_rows)

    output_path = save_dir / f"oat_summary_{period_start}_to_{period_end}.csv"
    oat_summary.to_csv(output_path, index=False)

    print(f"OAT summary saved to: {output_path}")

    return oat_results, oat_euis, oat_summary