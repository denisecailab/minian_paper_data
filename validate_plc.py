# %%
# imports and setup
import numpy as np
import xarray as xr
import pandas as pd
import seaborn as sns
import pickle as pkl
import place_cell as plc
import matplotlib.pyplot as plt
from matplotlib import ticker
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

SMALL_SIZE = 5
MEDIUM_SIZE = 6
BIG_SIZE = 7
sns.set(
    rc={
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica"],
        "font.size": MEDIUM_SIZE,
        "axes.titlesize": MEDIUM_SIZE,
        "axes.labelsize": MEDIUM_SIZE,  # size of faceting titles
        "xtick.labelsize": SMALL_SIZE,
        "ytick.labelsize": SMALL_SIZE,
        "legend.fontsize": MEDIUM_SIZE,
        "figure.titlesize": BIG_SIZE,
        "legend.edgecolor": "gray",
        "axes.linewidth": 0.4,
        "axes.facecolor": "white",
        "xtick.major.size": 2,
        "xtick.major.width": 0.4,
        "xtick.minor.visible": True,
        "xtick.minor.size": 1,
        "xtick.minor.width": 0.4,
        "ytick.major.size": 2,
        "ytick.major.width": 0.4,
        "ytick.minor.visible": True,
        "ytick.minor.size": 1,
        "ytick.minor.width": 0.4,
    }
)
sns.set_style("ticks")

# %%
# open datasets
plcds = xr.open_dataset("./data/inter/place_cells.nc")
fr = plcds["fr"].compute()
with open("./data/inter/mappings_pfd2.pkl", "rb") as fmap:
    match = pkl.load(fmap)
with open("./data/inter/mappings_sh_pfd2.pkl", "rb") as fmap:
    match_sh = pkl.load(fmap)

# %%
# transform mappings
def flatten_ss(row):
    ssA = row["group", "group"][0]
    ssB = row["group", "group"][1]
    return pd.Series(
        {
            "animal": row["meta", "animal"],
            "sessionA": ssA,
            "sessionB": ssB,
            "uidA": row["session", ssA],
            "uidB": row["session", ssB],
        }
    )


match = match[match["group", "group"].notnull()]
match_flt = match.apply(flatten_ss, axis="columns")
mask_si = plcds["mask_si"].to_dataframe()["mask_si"].astype(bool)
maxpos = plcds["maxpos"].to_dataframe()["maxpos"]
mask_sz = maxpos.notnull()
match_flt["mask_si_ssA"] = mask_si.loc[
    list(match_flt[["animal", "sessionA", "uidA"]].itertuples(index=False, name=None))
].values
match_flt["mask_si_ssB"] = mask_si.loc[
    list(match_flt[["animal", "sessionB", "uidB"]].itertuples(index=False, name=None))
].values
match_flt["mask_sz_ssA"] = mask_sz.loc[
    list(match_flt[["animal", "sessionA", "uidA"]].itertuples(index=False, name=None))
].values
match_flt["mask_sz_ssB"] = mask_sz.loc[
    list(match_flt[["animal", "sessionB", "uidB"]].itertuples(index=False, name=None))
].values
match_flt["maxpos"] = maxpos.loc[
    list(match_flt[["animal", "sessionA", "uidA"]].itertuples(index=False, name=None))
].values
match_flt = match_flt.sort_values("maxpos")
match_ft = match_flt.dropna()
match_ft = (
    match_flt[match_flt[["mask_sz_ssA", "mask_sz_ssB"]].all(axis="columns")]
    .dropna()
    .reset_index()
)

# %%
# compute correlations on different shifts
def corr(df):
    fr0 = fr.sel(
        animal=df["animal"].to_xarray(),
        session=df["sessionA"].to_xarray(),
        unit_id=df["uidA"].to_xarray(),
    )
    fr1 = fr.sel(
        animal=df["animal"].to_xarray(),
        session=df["sessionB"].to_xarray(),
        unit_id=df["uidB"].to_xarray(),
    )
    corrs = plc.vec_corr(fr0, fr1)
    return corrs.mean("index").values


match_flt_sh = (
    match_sh.dropna()
    .droplevel(0, axis="columns")
    .rename({"s10": "uidA", "s11": "uidB"}, axis="columns")
)
match_flt_sh["sessionA"] = "s10"
match_flt_sh["sessionB"] = "s11"
maxpos = plcds["maxpos"].to_dataframe()["maxpos"]
mask_sz = maxpos.notnull()
match_flt_sh["mask_sz_ssA"] = mask_sz.loc[
    list(
        match_flt_sh[["animal", "sessionA", "uidA"]].itertuples(index=False, name=None)
    )
].values
match_flt_sh["mask_sz_ssB"] = mask_sz.loc[
    list(
        match_flt_sh[["animal", "sessionB", "uidB"]].itertuples(index=False, name=None)
    )
].values
match_ft_sh = match_flt_sh.dropna()
match_ft_sh = match_flt_sh[
    match_flt_sh[["mask_sz_ssA", "mask_sz_ssB"]].all(axis="columns")
].dropna()
corrs = match_ft_sh.groupby(["hshift", "wshift"]).apply(corr)
corrs = corrs.rename("corr").astype(float).reset_index()
corrs.to_pickle("./data/inter/corrs_sh_pfd2.pkl")
# %%
# generate plot
def plc_heatmap(data, ax, **kwargs):
    sns.heatmap(data, ax=ax, cmap="viridis", rasterized=True, **kwargs)
    ax.invert_yaxis()
    ax.set_xlabel("Spatial Bin", fontstyle="italic")
    ax.set_ylabel("Cell ID", fontstyle="italic")
    ax.set_xticks(np.arange(0, 201, 50) + 0.5)
    ax.set_xticklabels(np.arange(0, 201, 50), rotation="horizontal", ha="center")
    ax.set_xlim((0, data.shape[1]))
    ax.set_yticks(np.arange(0, len(data) + 1, 50) + 0.5)
    ax.set_yticklabels(np.arange(0, len(data) + 1, 50), va="center")
    ax.set_ylim((0, data.shape[0]))
    ax.margins(0)
    for sp in ["top", "bottom", "left", "right"]:
        ax.spines[sp].set_visible(True)
        ax.spines[sp].set_color("black")
        ax.spines[sp].set_linewidth(0.4)


def corr_heatmap(data, ax, **kwargs):
    sns.heatmap(data, ax=ax, rasterized=True, **kwargs)
    ax.set_xlabel("Horizontol Shifts (px)", fontstyle="italic")
    ax.set_ylabel("Vertical Shifts (px)", fontstyle="italic")
    ax.set_xticks(np.arange(0, 101, 10) + 0.5)
    ax.set_xticklabels(np.arange(-50, 51, 10), rotation="horizontal", ha="center")
    ax.set_xlim((0, data.shape[1]))
    ax.set_yticks(np.arange(0, 101, 10) + 0.5)
    ax.set_yticklabels(np.arange(-50, 51, 10), va="center")
    ax.set_ylim((0, data.shape[0]))
    ax.margins(0)
    for sp in ["top", "bottom", "left", "right"]:
        ax.spines[sp].set_visible(True)
        ax.spines[sp].set_color("black")
        ax.spines[sp].set_linewidth(0.4)


aspect = 0.7
fig = plt.figure(constrained_layout=True)
fig.set_dpi(500)
fig.set_size_inches((5.31, 5.31 / aspect))
gs0 = fig.add_gridspec(2, 1, height_ratios=(1.6, 1))
gs1 = gs0[0].subgridspec(1, 2)
ax_ssA = fig.add_subplot(gs1[0, 0])
ax_ssA.set_title("Session 1", fontweight="bold")
ax_ssB = fig.add_subplot(gs1[0, 1])
ax_ssB.set_title("Session 2", fontweight="bold")
ax_corr = fig.add_subplot(gs0[1, 0])
ax_corr.set_title("Correlations", fontweight="bold")
plc_heatmap(
    fr.sel(
        animal=match_ft["animal"].to_xarray(),
        session=match_ft["sessionA"].to_xarray(),
        unit_id=match_ft["uidA"].to_xarray(),
    )
    .to_dataframe()
    .reset_index()
    .pivot("index", "x_bins", "fr"),
    ax_ssA,
    cbar=False,
)
plc_heatmap(
    fr.sel(
        animal=match_ft["animal"].to_xarray(),
        session=match_ft["sessionB"].to_xarray(),
        unit_id=match_ft["uidB"].to_xarray(),
    )
    .to_dataframe()
    .reset_index()
    .pivot("index", "x_bins", "fr"),
    ax_ssB,
    cbar=False,
)
corr_heatmap(
    corrs.pivot("hshift", "wshift", "corr"),
    ax_corr,
    cmap="RdBu_r",
    cbar=False,
    center=0,
    square=True,
)
ax_corr_cbar = inset_axes(
    ax_corr,
    width="50%",
    height="80%",
    loc="center",
    bbox_to_anchor=(1, 0, 0.15, 1),
    bbox_transform=ax_corr.transAxes,
    borderpad=0,
)
corr_cbar = fig.colorbar(ax_corr.collections[0], cax=ax_corr_cbar, extend="both")
corr_cbar.minorticks_on()
fig.savefig("./figs/validate_plc.svg")
