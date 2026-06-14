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
