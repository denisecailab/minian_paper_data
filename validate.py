# %%
# imports and setup
import numpy as np
import xarray as xr
import pandas as pd
import pickle as pkl
import functools as fct
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.colors import hsv_to_rgb
from dask.array import tensordot
from place_cell import thres_gmm
from minian_snapshot.minian.utilities import open_minian, load_videos

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
hipp_ds = open_minian("./data/pfd2/ts45-4/s10", backend="zarr")
nac_ds = open_minian("./data/nac/SF2-3_pretest2_new", backend="zarr")
# %%
# generate index of calcium events
def count_smp(df):
    df["sample"] = np.arange(len(df))
    return df


def sample(df, fm_crd):
    fm = df.iloc[0].loc["frame"]
    idxs = fm_crd.sel(frame=slice(fm - 300, fm + 301)).values
    idxs = pd.Series(idxs, name="frame")
    idxs.index.name = "fm_sample"
    return idxs


hipp_S = xr.apply_ufunc(
    thres_gmm,
    hipp_ds["S"].compute(),
    input_core_dims=[["frame"]],
    output_core_dims=[["frame"]],
    vectorize=True,
)
nac_S = xr.apply_ufunc(
    thres_gmm,
    nac_ds["S"].compute(),
    input_core_dims=[["frame"]],
    output_core_dims=[["frame"]],
    vectorize=True,
)
hipp_evt = hipp_S.to_series().astype(bool).reset_index()
nac_evt = nac_S.to_series().astype(bool).reset_index()
hipp_evt = (
    hipp_evt[hipp_evt["S"]].reset_index(drop=True).groupby("unit_id").apply(count_smp)
)
nac_evt = (
    nac_evt[nac_evt["S"]].reset_index(drop=True).groupby("unit_id").apply(count_smp)
)
hipp_idxs = (
    hipp_evt.groupby(["unit_id", "sample"])
    .apply(fct.partial(sample, fm_crd=hipp_S.coords["frame"]))
    .reset_index()
)
nac_idxs = (
    nac_evt.groupby(["unit_id", "sample"])
    .apply(fct.partial(sample, fm_crd=nac_S.coords["frame"]))
    .reset_index()
)
hipp_idxs.to_pickle("./data/inter/hipp_idxs.pkl")
nac_idxs.to_pickle("./data/inter/nac_idxs.pkl")

# %%
# open index of calcium events
with open("./data/inter/hipp_idxs.pkl", "rb") as pklf:
    hipp_idxs = pkl.load(pklf)
with open("./data/inter/nac_idxs.pkl", "rb") as pklf:
    nac_idxs = pkl.load(pklf)

# %%
# compute ya samples
def shuf_smp(df, fm_crd, sh_ext):
    sh = np.random.randint(sh_ext)
    fm_sh = fm_crd.roll({"frame": sh}, roll_coords=False).to_series()
    df["frame"] = df["frame"].map(fm_sh)
    return df


hipp_idxs_shuf = hipp_idxs.groupby("unit_id").apply(
    fct.partial(shuf_smp, fm_crd=hipp_ds.coords["frame"], sh_ext=hipp_ds.sizes["frame"])
)
nac_idxs_shuf = nac_idxs.groupby("unit_id").apply(
    fct.partial(shuf_smp, fm_crd=nac_ds.coords["frame"], sh_ext=nac_ds.sizes["frame"])
)
hipp_YA = (
    xr.apply_ufunc(
        tensordot,
        hipp_ds["Y"],
        hipp_ds["A"],
        input_core_dims=[["frame", "height", "width"], ["height", "width", "unit_id"]],
        output_core_dims=[["frame", "unit_id"]],
        dask="allowed",
        kwargs={"axes": [(1, 2), (0, 1)]},
    )
    .compute()
    .rename("ya")
    .to_series()
    .reset_index()
)
nac_YA = (
    xr.apply_ufunc(
        tensordot,
        nac_ds["Y"],
        nac_ds["A"],
        input_core_dims=[["frame", "height", "width"], ["height", "width", "unit_id"]],
        output_core_dims=[["frame", "unit_id"]],
        dask="allowed",
        kwargs={"axes": [(1, 2), (0, 1)]},
    )
    .compute()
    .rename("ya")
    .to_series()
    .reset_index()
)
hipp_idxs["fm_sample"] = hipp_idxs["fm_sample"] - 150
hipp_idxs_shuf["fm_sample"] = hipp_idxs_shuf["fm_sample"] - 150
nac_idxs["fm_sample"] = nac_idxs["fm_sample"] - 150
nac_idxs_shuf["fm_sample"] = nac_idxs_shuf["fm_sample"] - 150
hipp_yathres = (
    hipp_idxs.merge(hipp_YA, on=["unit_id", "frame"])
    .groupby(["unit_id", "fm_sample"])["ya"]
    .mean()
    .reset_index()
)
hipp_yathres_shuf = (
    hipp_idxs_shuf.merge(hipp_YA, on=["unit_id", "frame"])
    .groupby(["unit_id", "fm_sample"])["ya"]
    .mean()
    .reset_index()
)
nac_yathres = (
    nac_idxs.merge(nac_YA, on=["unit_id", "frame"])
    .groupby(["unit_id", "fm_sample"])["ya"]
    .mean()
    .reset_index()
)
nac_yathres_shuf = (
    nac_idxs_shuf.merge(nac_YA, on=["unit_id", "frame"])
    .groupby(["unit_id", "fm_sample"])["ya"]
    .mean()
    .reset_index()
)
hipp_yathres["shuf"] = "Aligned"
hipp_yathres_shuf["shuf"] = "Shuffled"
nac_yathres["shuf"] = "Aligned"
nac_yathres_shuf["shuf"] = "Shuffled"
hipp_ya = pd.concat([hipp_yathres, hipp_yathres_shuf], ignore_index=True)
nac_ya = pd.concat([nac_yathres, nac_yathres_shuf], ignore_index=True)

# %%
# load raw video and compute max projections
def norm(a):
    amin = np.nanmin(a)
    amax = np.nanmax(a)
    return (a - amin) / (amax - amin)


param_load_videos = {
    "pattern": r"msCam[0-9]+\.avi$",
    "dtype": np.uint8,
    "downsample_strategy": "subset",
}
hipp_varr = load_videos(
    "./data/pfd2/ts45-4/s10",
    downsample={"frame": 2, "height": 2, "width": 2},
    **param_load_videos,
)
nac_varr = load_videos(
    "./data/nac/SF2-3_pretest2_new",
    downsample={"frame": 2, "height": 1, "width": 1},
    **param_load_videos,
)
hipp_max_raw = hipp_varr.max("frame").compute()
hipp_max_y = hipp_ds["Y"].max("frame").compute()
nac_max_raw = nac_varr.max("frame").compute()
nac_max_y = nac_ds["Y"].max("frame").compute()
hipp_a = xr.apply_ufunc(
    norm,
    hipp_ds["A"].compute(),
    input_core_dims=[["height", "width"]],
    output_core_dims=[["height", "width"]],
    vectorize=True,
)
nac_a = xr.apply_ufunc(
    norm,
    nac_ds["A"].compute(),
    input_core_dims=[["height", "width"]],
    output_core_dims=[["height", "width"]],
    vectorize=True,
)

# %%
# compute shuffled correlations
def roll2(A, hsh, wsh):
    return np.roll(np.roll(A, hsh, axis=0), wsh, axis=1)


def corr_roll(A, max_proj):
    corr_ls = []
    for _ in range(1000):
        hsh = np.random.randint(0, A.sizes["height"], A.sizes["unit_id"])
        wsh = np.random.randint(0, A.sizes["width"], A.sizes["unit_id"])
        Ash = xr.apply_ufunc(
            roll2,
            A,
            hsh,
            wsh,
            input_core_dims=[["height", "width"], [], []],
            output_core_dims=[["height", "width"]],
            vectorize=True,
        )
        Ash_sum = Ash.sum("unit_id")
        corr = np.corrcoef(Ash_sum.values.reshape(-1), max_proj.values.reshape(-1))
        corr_ls.append(corr)
    return np.array(corr_ls)[:, 0, 1]


hipp_corrs = corr_roll(hipp_ds["A"].compute(), hipp_max_y)
hipp_corr_true = np.corrcoef(
    hipp_ds["A"].sum("unit_id").values.reshape(-1), hipp_max_y.values.reshape(-1)
)[0, 1]
nac_corrs = corr_roll(nac_ds["A"].compute(), nac_max_y)
nac_corr_true = np.corrcoef(
    nac_ds["A"].sum("unit_id").values.reshape(-1), nac_max_y.values.reshape(-1)
)[0, 1]
np.save("./data/inter/hipp_corrs.npy", hipp_corrs)
np.save("./data/inter/nac_corrs.npy", nac_corrs)
# %%
# open shuffled correlations
hipp_corrs = np.load("./data/inter/hipp_corrs.npy")
nac_corrs = np.load("./data/inter/nac_corrs.npy")
# %%
# generate plot
def lineplot(data, ylim, ax):
    sns.lineplot(
        x="fm_sample",
        y="ya",
        hue="shuf",
        hue_order=["Shuffled", "Aligned"],
        data=data,
        ax=ax,
        linewidth=1,
        legend="full",
    )
    ax.axvline(0, color="k", linewidth=0.4, linestyle="--")
    ax.set_xlabel("Frame", fontstyle="italic")
    ax.set_ylabel("Signal (A.U.)", fontstyle="italic")
    len_hand, len_lab = ax.get_legend_handles_labels()
    ax.legend(handles=len_hand[1:], labels=len_lab[1:], loc="upper left")
    xlim = (-165, 165)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect((1 / np.diff(ylim)) / (1.5 / np.diff(xlim)))


def implot(arr, ax, **kwargs):
    if arr.ndim == 2:
        ax.imshow(arr, cmap="gray", **kwargs)
        ax.invert_yaxis()
    elif arr.ndim == 3:
        arr_h = (arr > 0) * np.random.random(arr.shape[0]).reshape((-1, 1, 1))
        arr_s = np.ones_like(arr)
        arr_v = arr
        arr_hsv = np.stack([arr_h, arr_s, arr_v], axis=-1)
        arr_rgb = hsv_to_rgb(arr_hsv).sum(axis=0)
        ax.imshow(arr_rgb, **kwargs)
        ax.invert_yaxis()
    ax.set_ylim((0, arr.shape[-1] * 480 / 752))
    ax.margins(0)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)


def corrplot(corrs, corr_true, ax, **kwargs):
    sns.distplot(
        corrs,
        bins=15,
        norm_hist=True,
        kde_kws={"linewidth": 1},
        hist_kws={"alpha": 0.7, "density": True, "rwidth": 1, "linewidth": 0.2},
        ax=ax,
        **kwargs,
    )
    ax.axvline(corr_true, linewidth=1, color="red")
    ax.set_xlabel("Correlation", fontstyle="italic")
    ax.set_ylabel("Count", fontstyle="italic")
    xlim = (-0.15, 0.7)
    ylim = (0, 18)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_yticks(np.arange(start=ylim[0], stop=ylim[1], step=3))
    ax.set_aspect((1 / np.diff(ylim)) / (1.4 / np.diff(xlim)))
    texprops = {
        "fontsize": MEDIUM_SIZE,
        "horizontalalignment": "center",
        "verticalalignment": "center",
    }
    arrowprops = {
        "arrowstyle": "->",
        "shrinkA": 2,
        "shrinkB": 14,
        "linewidth": 1,
        "edgecolor": "black",
    }
    ax.annotate(
        "Null\nDistribution",
        xy=(0, 6),
        xytext=(0.15, 16),
        arrowprops=arrowprops,
        **texprops,
    )
    ax.annotate(
        "Observed\nCorrelation",
        xy=(corr_true, 6),
        xytext=(corr_true - 0.15, 16),
        arrowprops=arrowprops,
        **texprops,
    )


aspect = 0.75
fig = plt.figure(constrained_layout=True)
fig.set_dpi(500)
fig.set_size_inches((5.31, 5.31 / aspect))
gs = fig.add_gridspec(4, 2, height_ratios=(1.1, 1, 1.1, 1))
ax_hipp_ymax = fig.add_subplot(gs[0, 0])
ax_hipp_a = fig.add_subplot(gs[0, 1])
ax_hipp_corr = fig.add_subplot(gs[1, 0])
ax_hipp_ya = fig.add_subplot(gs[1, 1])
ax_nac_ymax = fig.add_subplot(gs[2, 0])
ax_nac_a = fig.add_subplot(gs[2, 1])
ax_nac_corr = fig.add_subplot(gs[3, 0])
ax_nac_ya = fig.add_subplot(gs[3, 1])
implot(hipp_max_y.values, ax_hipp_ymax)
ax_hipp_ymax.set_title("Max Projection")
ax_hipp_ymax.text(
    -0.1,
    1.2,
    "A",
    fontsize=BIG_SIZE,
    fontweight="bold",
    transform=ax_hipp_ymax.transAxes,
)
implot(hipp_a.values, ax_hipp_a)
ax_hipp_a.set_title("Spatial Footprints")
corrplot(hipp_corrs, hipp_corr_true, ax_hipp_corr)
ax_hipp_corr.set_title(
    "Correlations Between Max Projection\nand Shuffled Spatial Footprints"
)
lineplot(
    data=hipp_ya, ylim=(10, 50), ax=ax_hipp_ya,
)
ax_hipp_ya.set_title("Fluorescence Signal\nRelative to Calcium Events")
implot(nac_max_y.values, ax_nac_ymax)
ax_nac_ymax.set_title("Max Projection")
ax_nac_ymax.text(
    -0.1,
    1.2,
    "B",
    fontsize=BIG_SIZE,
    fontweight="bold",
    transform=ax_nac_ymax.transAxes,
)
implot(nac_a.values, ax_nac_a)
ax_nac_a.set_title("Spatial Footprints")
corrplot(nac_corrs, nac_corr_true, ax_nac_corr)
ax_nac_corr.set_title(
    "Correlations Between Max Projection\nand Shuffled Spatial Footprints"
)
lineplot(
    data=nac_ya, ylim=(13, 45), ax=ax_nac_ya,
)
ax_nac_ya.set_title("Fluorescence Signal\nRelative to Calcium Events")
# fig.tight_layout()
fig.savefig("./figs/validate.svg")
