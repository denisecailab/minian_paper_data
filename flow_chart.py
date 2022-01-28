# %%
# imports and setup
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import xarray as xr
import pandas as pd
from numpy import random
from minian_snapshot.minian.utilities import open_minian, load_videos
from minian_snapshot.minian.initialization import seeds_init
from matplotlib.patches import ConnectionPatch
from matplotlib.colors import hsv_to_rgb
from bokeh.palettes import Category10_10

SMALL_SIZE = 8
MEDIUM_SIZE = 11
BIG_SIZE = 11
sns.set(
    rc={
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial"],
        "font.size": MEDIUM_SIZE,
        "axes.titlesize": BIG_SIZE,
        "axes.labelsize": SMALL_SIZE,  # size of faceting titles
        "xtick.labelsize": SMALL_SIZE,
        "ytick.labelsize": SMALL_SIZE,
        "legend.fontsize": SMALL_SIZE,
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
# load dataset and compute
def norm(a):
    amin = np.nanmin(a)
    amax = np.nanmax(a)
    return (a - amin) / (amax - amin)


param_load_videos = {
    "pattern": r"msCam[0-9]+\.avi$",
    "dtype": np.uint8,
    "downsample_strategy": "subset",
    "downsample": {"frame": 2, "height": 2, "width": 2},
}
param_seeds_init = {
    "wnd_size": 1000,
    "method": "rolling",
    "stp_size": 500,
    "nchunk": 100,
    "max_wnd": 15,
    "diff_thres": 3,
}

varr = load_videos("./data/pfd2/ts45-4/s10", **param_load_videos)
minian_ds = open_minian("./data/pfd2/ts45-4/s10", backend="zarr")
max_varr = varr.max("frame").compute()
max_Y = minian_ds["Y"].max("frame").compute()
shifts = (
    minian_ds["shifts"]
    .to_dataframe()
    .reset_index()
    .replace({"height": "Horizontal", "width": "Vertical"})
)
seeds = seeds_init(minian_ds["Y"], **param_seeds_init)
A = xr.apply_ufunc(
    norm,
    minian_ds["A"].compute(),
    input_core_dims=[["height", "width"]],
    output_core_dims=[["height", "width"]],
    vectorize=True,
)
# units = random.choice(minian_ds.coords["unit_id"], size=10, replace=False)
units = [260, 348, 390, 318, 197, 365, 230, 440, 271, 71]
C = minian_ds["C"].sel(unit_id=units).rename("val").to_dataframe().reset_index()
S = minian_ds["S"].sel(unit_id=units).rename("val").to_dataframe().reset_index()
C["trace"] = "Calcium"
S["trace"] = "Spikes"
tmp_df = pd.concat([C, S], ignore_index=True)

# %%
# generate plot
## define functions and constants
def implot(arr, ax, **kwargs):
    if arr.ndim == 2:
        ax.imshow(arr, cmap="viridis", **kwargs)
        ax.invert_yaxis()
    elif arr.ndim == 3:
        arr_h = (arr > 0) * random.random(arr.shape[0]).reshape((-1, 1, 1))
        arr_s = np.ones_like(arr)
        arr_v = arr
        arr_hsv = np.stack([arr_h, arr_s, arr_v], axis=-1)
        arr_rgb = hsv_to_rgb(arr_hsv).sum(axis=0)
        ax.imshow(arr_rgb, **kwargs)
        ax.invert_yaxis()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)


def shplot(x, y, data, hue, ax, xlim, ylim, **kwargs):
    sns.lineplot(
        x=x,
        y=y,
        data=data,
        hue=hue,
        ax=ax,
        linewidth=0.8,
        legend="full",
        palette={"Horizontal": Category10_10[2], "Vertical": Category10_10[4]},
        **kwargs,
    )
    ax.set_xlabel("Frame", labelpad=0.3, fontstyle="italic")
    ax.set_ylabel("Shift (px)", labelpad=0.3, fontstyle="italic")
    # len_hand, len_lab = ax.get_legend_handles_labels()
    # ax.legend(handles=len_hand[1:], labels=len_lab[1:], loc="upper right")
    ax.legend(loc="upper right")
    ax.set_aspect((480 / np.diff(ylim)) / (752 / np.diff(xlim)))
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xticks(np.arange(0, xlim[1] + 1, 200))
    ax.set_yticks(np.arange(-4, ylim[1] + 1, 2))
    ax.minorticks_off()
    ax.tick_params(pad=0.3)
    ax.spines["left"].set_position(("data", 0))
    ax.spines["left"].set_bounds(-4, ylim[1])
    ax.spines["right"].set_bounds(-4, ylim[1])
    ax.spines["bottom"].set_position(("data", -4))
    ax.spines["bottom"].set_bounds(0, xlim[1])
    ax.spines["top"].set_bounds(0, xlim[1])


def tmplot(df, offset, xlim, ylim, ax, **kwargs):
    df = df[df["frame"].between(*xlim)].copy()
    df["val_norm"] = df.groupby(["unit_id", "trace"])["val"].apply(norm)
    df["val_norm"] = df.apply(
        lambda r: r["val_norm"] * 0.5 if r["trace"] == "Spikes" else r["val_norm"],
        axis="columns",
    )
    for igrp, (uid, df_sub) in enumerate(df.groupby("unit_id", sort=False)):
        df_sub = df_sub.copy()
        df_sub["val_norm"] = df_sub["val_norm"] + igrp * offset
        sns.lineplot(
            x="frame",
            y="val_norm",
            data=df_sub,
            hue="trace",
            hue_order=["Spikes", "Calcium"],
            # size="trace",
            linewidth=0.4,
            ax=ax,
            legend="brief",
            zorder=-igrp,
            palette={"Spikes": Category10_10[0], "Calcium": Category10_10[1]},
            # sizes={"Spikes": 0.3, "Calcium": 0.5},
            **kwargs,
        )
        caldf = df_sub[df_sub["trace"] == "Calcium"]
        ycal = caldf["val_norm"].values
        ax.fill_between(
            x=caldf["frame"],
            y1=ycal,
            y2=np.ones_like(ycal) * igrp * offset,
            facecolor="white",
            zorder=-igrp,
        )
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.get_yaxis().set_visible(False)
    ax.spines["bottom"].set_position(("data", 0))
    ax.spines["left"].set_bounds(0, ylim[1])
    ax.spines["right"].set_bounds(0, ylim[1])
    ax.set_xlabel("Frame", labelpad=0.3, fontstyle="italic")
    ax.tick_params(pad=0.3)
    ax.minorticks_off()
    len_hand, len_lab = ax.get_legend_handles_labels()
    ax.legend(handles=len_hand[1:3], labels=len_lab[1:3], loc="upper right")
    ax.set_aspect((480 / np.diff(ylim)) / (752 / np.diff(xlim)))


arrowstyle = {
    "arrowstyle": "simple, head_length=1, head_width=2, tail_width=1",
    "linewidth": 0.6,
    "facecolor": "gray",
    "edgecolor": "black",
    "capstyle": "round",
    "joinstyle": "bevel",
    "shrinkA": 4,
    "shrinkB": 4,
}
## setup figure
aspect = 2
fig, axs = plt.subplots(2, 3, figsize=(7.87, 7.87 / aspect), dpi=500)
ax_raw = axs[0, 0]
ax_Y = axs[0, 1]
ax_sh = axs[0, 2]
ax_sd = axs[1, 2]
ax_sp = axs[1, 1]
ax_tmp = axs[1, 0]
## plotting
implot(max_varr.values, ax=ax_raw)
implot(max_Y.values, ax=ax_Y)
shplot(
    x="frame",
    y="shifts",
    data=shifts,
    hue="variable",
    ax=ax_sh,
    xlim=(-120, 800),
    ylim=(-6, 7),
)
implot(max_Y.values, ax=ax_sd)
ax_sd.scatter(
    seeds["width"].values / 2,
    seeds["height"].values / 2,
    s=0.1,
    linewidths=0.1,
    c="silver",
    marker=",",
)
implot(A.values, ax=ax_sp)
tmplot(tmp_df, xlim=(0, 5000), ylim=(-1, 6), offset=0.5, ax=ax_tmp)
## add titles
ax_raw.text(
    x=0.5,
    y=1.05,
    s="Raw Video",
    transform=ax_raw.transAxes,
    fontweight="bold",
    fontsize=BIG_SIZE,
    horizontalalignment="center",
    verticalalignment="bottom",
)
ax_Y.text(
    x=0.5,
    y=1.05,
    s="Preprocessing",
    transform=ax_Y.transAxes,
    fontweight="bold",
    fontsize=BIG_SIZE,
    horizontalalignment="center",
    verticalalignment="bottom",
)
ax_sh.text(
    x=0.5,
    y=1.05,
    s="Motion Correction",
    transform=ax_sh.transAxes,
    fontweight="bold",
    fontsize=BIG_SIZE,
    horizontalalignment="center",
    verticalalignment="bottom",
)
ax_sd.text(
    x=0.5,
    y=-0.05,
    s="Seeds Initialization",
    transform=ax_sd.transAxes,
    fontweight="bold",
    fontsize=BIG_SIZE,
    horizontalalignment="center",
    verticalalignment="top",
)
ax_sp.text(
    x=0.5,
    y=-0.05,
    s="Spatial Update",
    transform=ax_sp.transAxes,
    fontweight="bold",
    fontsize=BIG_SIZE,
    horizontalalignment="center",
    verticalalignment="top",
)
ax_tmp.text(
    x=0.5,
    y=-0.05,
    s="Temporal Update",
    transform=ax_tmp.transAxes,
    fontweight="bold",
    fontsize=BIG_SIZE,
    horizontalalignment="center",
    verticalalignment="top",
)
## add arrows
con1 = ConnectionPatch(
    xyA=(1, 0.5),
    xyB=(0, 0.5),
    coordsA="axes fraction",
    coordsB="axes fraction",
    axesA=ax_raw,
    axesB=ax_Y,
    **arrowstyle,
)
con2 = ConnectionPatch(
    xyA=(1, 0.5),
    xyB=(0, 0.5),
    coordsA="axes fraction",
    coordsB="axes fraction",
    axesA=ax_Y,
    axesB=ax_sh,
    **arrowstyle,
)
con3 = ConnectionPatch(
    xyA=(0.5, 0),
    xyB=(0.5, 1),
    coordsA="axes fraction",
    coordsB="axes fraction",
    axesA=ax_sh,
    axesB=ax_sd,
    **arrowstyle,
)
con4 = ConnectionPatch(
    xyA=(0, 0.5),
    xyB=(1, 0.5),
    coordsA="axes fraction",
    coordsB="axes fraction",
    axesA=ax_sd,
    axesB=ax_sp,
    **arrowstyle,
)
con5 = ConnectionPatch(
    xyA=(0, 0.5),
    xyB=(1, 0.5),
    coordsA="axes fraction",
    coordsB="axes fraction",
    axesA=ax_sp,
    axesB=ax_tmp,
    **arrowstyle,
)
con6 = ConnectionPatch(
    xyA=(0.8, 1),
    xyB=(0.2, 1),
    coordsA="axes fraction",
    coordsB="axes fraction",
    axesA=ax_tmp,
    axesB=ax_sp,
    connectionstyle="arc3, rad=-0.4",
    arrowstyle="simple, head_length=1, head_width=2, tail_width=1",
    linewidth=0.6,
    facecolor="gray",
    edgecolor="black",
    capstyle="round",
    joinstyle="bevel",
    shrinkA=17,
    shrinkB=17,
)
ax_raw.add_artist(con1)
ax_Y.add_artist(con2)
ax_sh.add_artist(con3)
ax_sd.add_artist(con4)
ax_sp.add_artist(con5)
fig.tight_layout(w_pad=1.5, h_pad=4)
ax_tmp.add_artist(con6)
plt.annotate(
    "Repeat",
    (0.5, 1),
    xycoords=con6,
    fontweight="bold",
    fontsize=BIG_SIZE,
    horizontalalignment="center",
    verticalalignment="bottom",
)
fig.savefig("./figs/pipeline.svg")
fig.savefig("./figs/pipeline.tiff")
fig.savefig("./figs/pipeline.png")
