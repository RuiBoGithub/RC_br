# [markdown]
#  POST

LABEL_MAP = {
    "a5_infiltration_rate":
        r"Infiltration Rate ($\mathrm{m^{3}\cdot h^{-3}\cdot h^{-1}}$)",

    "d5_infiltration_rate_atrium":
        r"Air Change Rate (Atrium) ($\mathrm{h^{-1}}$)",

    "b12_sat":
        r"Supply Air Temperature (AHU) (°C)",

    "d1_htgsp_office":
        r"Heating Setpoint Temperature (°C)",

    "e1_natural_ventilation_rate":
        r"Window Opening Rate (-)",

    "b10_airloophvac":
        r"Return Air Flow Fraction (-)",
}

import matplotlib.pyplot as plt
plt.rcParams.update({
    'font.family': 'Arial',"mathtext.fontset": "stix",
    'font.size': 14,
    'axes.labelsize': 18,
    'legend.fontsize': 12,
})
from pathlib import Path
import matplotlib.pyplot as plt

def save_and_show(fig, out_dir, name, dpi=300):
    """
    Save figure safely and fully release matplotlib state.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    path = out_dir / f"{name}.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    print(f"✓ Saved: {path}")

    plt.show()
    plt.close(fig)

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import pandas as pd
from pathlib import Path
import subprocess
import os
import math

PROJECT_ROOT = Path(
    os.getenv("PROJECT_ROOT", Path.cwd())
).resolve()
PROJECT_PARENT = PROJECT_ROOT.parent

identifier     = f'_{DATE}' 

PROJECT_ROOT = Path(
    os.getenv("PROJECT_ROOT", Path.cwd())
).resolve()
PROJECT_PARENT = PROJECT_ROOT.parent
from pathlib import Path
import subprocess
data_dir = Path(f"_json/_run/{DATE}/BC").resolve()
output_dir = Path(f"_json/_run/{DATE}/calibro").resolve()
output_dir.mkdir(parents=True, exist_ok=True)
code_dir = Path(f"_json/_run/{DATE}/_code")
code_dir.mkdir(parents=True, exist_ok=True)
CALIBRO_NAME = "simplest"
RSCRIPT_BIN = "Rscript"
# =====================================================
# 0. Paths
# =====================================================
Z_PATH           = Path(output_dir) / "posterior_draws_Z.csv"
TT_PATH          = Path(data_dir)   / "TT.csv"
THETAMAP_PATH    = Path(output_dir) / "theta_physical_summary.csv"

OUT_THETA_ALL    = Path(output_dir) / "posterior_draws_theta.csv"
OUT_THETA_TOP1   = Path(output_dir) / "posterior_top1_theta.csv"
OUT_THETA_TOP100P = Path(output_dir) / "posterior_top100pct_theta_map.csv"
OUT_THETA_RAND10 = Path(output_dir) / "posterior_rand10_from_topfrac.csv"


# =====================================================
# 1. Load data
# =====================================================
Z = pd.read_csv(Z_PATH)                     # posterior samples (Z-space)
TT = pd.read_csv(TT_PATH)                   # physical bounds
theta_map_df = pd.read_csv(THETAMAP_PATH)   # MAP estimates (physical)

# =====================================================
# 2. Clean parameter names
# =====================================================
theta_map_df["PARAMETER"] = (
    theta_map_df["PARAMETER"].astype(str).str.strip('"')
)

# =====================================================
# 3. Detect and order parameters (canonical order)
# =====================================================
physical_params = sorted(
    set(Z.columns) & set(TT.columns) & set(theta_map_df["PARAMETER"]),
    key=lambda x: (x[0], x)
)

# =====================================================
# 4. Construct MAP vector (physical space)
# =====================================================
theta_map = (
    theta_map_df
    .set_index("PARAMETER")
    .loc[physical_params, "ESTIMATE"]
)

# =====================================================
# 5. Z → Theta transform (physical space)
# =====================================================
tt_min = TT[physical_params].min()
tt_max = TT[physical_params].max()

Theta = tt_min + Z[physical_params] * (tt_max - tt_min)
# Theta.to_csv(OUT_THETA_ALL, index=False)

# =====================================================
# 6. Distance to MAP (Mahalanobis, covariance-aware)
# =====================================================
theta_range = tt_max - tt_min

# normalised θ-space
Theta_norm = (Theta - theta_map) / theta_range

# covariance + inverse
cov = np.cov(Theta_norm.T)
inv_cov = np.linalg.pinv(cov)

# Mahalanobis distance
dist = np.sqrt(
    np.einsum(
        "ij,jk,ik->i",
        Theta_norm.values,
        inv_cov,
        Theta_norm.values,
    )
)

# make dist a pandas Series (so nsmallest works)
dist = pd.Series(dist, index=Theta.index, name="dist_to_MAP")

# =====================================================
# 7. TOP-1 = MAP
# =====================================================
Theta_top1 = theta_map.to_frame().T.copy()
Theta_top1["dist_to_MAP"] = 0.0

# =====================================================
# 8a. TOP-fraction: 100% posterior draws
# =====================================================
top_frac_100 = 1.00

k_100 = max(1, int(len(dist) * top_frac_100))
top_idx_100 = dist.nsmallest(k_100).index

Theta_top100pc = Theta.loc[top_idx_100].copy()
Theta_top100pc["dist_to_MAP"] = dist.loc[top_idx_100].values

# prepend MAP row
Theta_map_row_100 = theta_map.to_frame().T.copy()
Theta_map_row_100["dist_to_MAP"] = 0.0

Theta_top100pc = pd.concat(
    [Theta_map_row_100, Theta_top100pc],
    ignore_index=True
)

Theta_top100pc.to_csv(OUT_THETA_TOP100P, index=False)

print(f"Saved {len(Theta_top100pc)} rows to:")
print(OUT_THETA_TOP100P)

# =====================================================
# 8b. TOP-fraction (nearest posterior draws)
# =====================================================
top_frac = 0.50   # 50%
RANDOM_N = 9

k = max(1, int(len(dist) * top_frac))
top_idx = dist.nsmallest(k).index

Theta_top10pc = Theta.loc[top_idx].copy()
Theta_top10pc["dist_to_MAP"] = dist.loc[top_idx].values

# prepend MAP row
Theta_map_row = theta_map.to_frame().T
Theta_map_row["dist_to_MAP"] = 0.0

Theta_top10pc = pd.concat(
    [Theta_map_row, Theta_top10pc],
    ignore_index=True
)

# =====================================================
# 8c. RANDOMLY DRAW 10 (from the top-fraction pool, excluding MAP)
# =====================================================
RANDOM_SEED = 42
pool = Theta_top10pc.iloc[1:].copy()  # exclude MAP row (row 0)
n_draw = min(RANDOM_N, len(pool))

Theta_rand10 = pool.sample(n=n_draw, random_state=RANDOM_SEED)

# include MAP back for plotting/saving
Theta_rand10 = pd.concat([Theta_top10pc.iloc[[0]], Theta_rand10], ignore_index=True)
Theta_rand10.to_csv(OUT_THETA_RAND10, index=False)

# =====================================================
# 9. KDE contour plotting (UNCHANGED except points source)
# =====================================================
CONTOUR_DOMAIN = "full"  # "full" or "Theta_top10pc"

if CONTOUR_DOMAIN == "Theta_top10pc":
    Theta_kde = Theta_top10pc
    domain_tag = "Theta_top10pc"
elif CONTOUR_DOMAIN == "full":
    Theta_kde = Theta
    domain_tag = "full"
else:
    raise ValueError("Invalid CONTOUR_DOMAIN")

y_sel = "d1_htgsp_office"
xs = [c for c in Theta.columns if c != y_sel]

fig, axes = plt.subplots(
    1, len(xs),
    figsize=(6 * len(xs), 6),
    constrained_layout=True,
    sharey=True,
)

if len(xs) == 1:
    axes = [axes]

for i, (ax, x) in enumerate(zip(axes, xs)):

    X = Theta_kde[[x, y_sel]].values.T
    kde = gaussian_kde(X)

    xi, yi = np.mgrid[
        X[0].min():X[0].max():200j,
        X[1].min():X[1].max():200j,
    ]
    zi = kde(np.vstack([xi.ravel(), yi.ravel()]))
    z = zi.reshape(xi.shape)

    levels = np.linspace(z.min(), z.max(), 15)
    levels = np.concatenate(([levels[0]], levels[5:]))

    ax.contourf(
        xi, yi, z,
        levels=levels,
        cmap="YlOrRd",
        alpha=0.75,
    )

    # ✅ random-10 points (including MAP row at index 0)
    ax.scatter(
        Theta_rand10[x],
        Theta_rand10[y_sel],
        s=25,
        c="black",
        alpha=0.35,
        zorder=3,
    )

    # MAP (still plotted explicitly in red)
    ax.scatter(
        Theta_top1[x],
        Theta_top1[y_sel],
        s=50,
        c="red",
        edgecolors="black",
        zorder=4,
    )

    ax.set_xlim(X[0].min(), X[0].max())
    ax.set_ylim(X[1].min(), X[1].max())

    ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))

    ax.set_xlabel(LABEL_MAP.get(x, x))
    ax.set_box_aspect(1)

    if i == 0:
        ax.set_ylabel(LABEL_MAP.get(y_sel, y_sel))
    else:
        ax.tick_params(left=False, labelleft=False)

out_name = f"_posterior_prior_{identifier}_KDE_{domain_tag}_rand10"
save_and_show(fig, out_dir= output_dir / f"_plt_{DATE}", name=out_name)


