# %%
import os
import sys
import re
import numpy as np
import xarray as xr
import pandas as pd
import holoviews as hv
import pickle as pkl
import place_cell as plc
from tqdm import tqdm_notebook
from dask.array import tensordot
from holoviews.operation.datashader import datashade, regrid
from scipy.signal import medfilt
from holoviews.util import Dynamic
from dask.diagnostics import ProgressBar
from natsort import natsorted
from sklearn.mixture import GaussianMixture
from scipy.ndimage.measurements import center_of_mass
from scipy.ndimage import label
from place_cell import thres_gmm
from minian_snapshot.minian.cnmf import compute_trace
from minian_snapshot.minian.utilities import open_minian, load_videos, get_optimal_chk

# %%
minian_ds = open_minian("./data/ts44-3/s10/", backend="zarr")

# %%
S = minian_ds["S"].compute()
S_thres = xr.apply_ufunc(
    thres_gmm,
    S,
    input_core_dims=[["frame"]],
    output_core_dims=[["frame"]],
    vectorize=True,
)

# %%
cevt = S_thres.to_series().astype(bool).reset_index()
cevt = cevt[cevt["S"]].reset_index(drop=True)


def count_smp(df):
    df["sample"] = np.arange(len(df))
    return df


cevt = cevt.groupby("unit_id").apply(count_smp)

# %%time
fm_crd = S_thres.coords["frame"]


def sample(df):
    fm = df.iloc[0].loc["frame"]
    idxs = fm_crd.sel(frame=slice(fm - 300, fm + 301)).values
    idxs = pd.Series(idxs, name="frame")
    idxs.index.name = "fm_sample"
    return idxs


idxs = cevt.groupby(["unit_id", "sample"]).apply(sample).reset_index()
idxs.to_pickle("./idxs.pkl")

# %%
with open("./idxs.pkl", "rb") as pklf:
    idxs = pkl.load(pklf)

# %%
fm_crd = S_thres.coords["frame"]
sh_ext = S_thres.sizes["frame"]


def shuf_smp(df):
    sh = np.random.randint(sh_ext)
    fm_sh = fm_crd.roll({"frame": sh}, roll_coords=False).to_series()
    df["frame"] = df["frame"].map(fm_sh)
    return df


idxs_shuf = idxs.groupby("unit_id").apply(shuf_smp)

# %%
# %%time
YA = xr.apply_ufunc(
    tensordot,
    minian_ds["Y"],
    minian_ds["A"],
    input_core_dims=[["frame", "height", "width"], ["height", "width", "unit_id"]],
    output_core_dims=[["frame", "unit_id"]],
    dask="allowed",
    kwargs={"axes": [(1, 2), (0, 1)]},
)
YA = YA.compute()

# %%
ya_df = YA.rename("ya").to_series().reset_index()
c_df = minian_ds["C"].to_series().reset_index()

# %%
idxs_unq = (
    idxs.groupby("unit_id")
    .apply(lambda df: df.drop_duplicates("frame")["frame"])
    .reset_index("unit_id")
)
idxs_unq["drop"] = True

# %%
ya_df_drop = ya_df.merge(idxs_unq, on=["unit_id", "frame"], how="left")
ya_df_drop = ya_df_drop[ya_df_drop["drop"].isnull()]

# %%
# %%time
ya_sthres = idxs.merge(ya_df, on=["unit_id", "frame"]).sort_values(
    ["unit_id", "sample", "fm_sample"]
)
c_sthres = idxs.merge(c_df, on=["unit_id", "frame"]).sort_values(
    ["unit_id", "sample", "fm_sample"]
)
ya_shuf = idxs_shuf.merge(ya_df, on=["unit_id", "frame"]).sort_values(
    ["unit_id", "sample", "fm_sample"]
)
ya_shuf_drop = idxs_shuf.merge(
    ya_df_drop, on=["unit_id", "frame"], how="left"
).sort_values(["unit_id", "sample", "fm_sample"])

# %%
mean_sthres = ya_sthres.groupby(["unit_id", "fm_sample"])["ya"].mean()
sum_sthres = mean_sthres.groupby("fm_sample").agg(["mean", "sem"])
meanc_sthres = c_sthres.groupby(["unit_id", "fm_sample"])["C"].mean()
sumc_sthres = meanc_sthres.groupby("fm_sample").agg(["mean", "sem"])
mean_shuf = ya_shuf.groupby(["unit_id", "fm_sample"])["ya"].mean()
sum_shuf = mean_shuf.groupby("fm_sample").agg(["mean", "sem"])
mean_shuf_drop = ya_shuf_drop.groupby(["unit_id", "fm_sample"])["ya"].mean()
sum_shuf_drop = mean_shuf_drop.groupby("fm_sample").agg(["mean", "sem"])

# %%
hvres = (
    hv.Curve(sum_sthres, "fm_sample", "mean").opts(color="black", title="Aligned")
    * hv.Spread(sum_sthres, "fm_sample", ["mean", "sem"])
    + hv.Curve(sumc_sthres, "fm_sample", "mean").opts(color="black", title="Shuffled")
    * hv.Spread(sum_shuf, "fm_sample", ["mean", "sem"])
    + hv.Curve(sum_shuf, "fm_sample", "mean").opts(color="black", title="Shuffled")
    * hv.Spread(sum_shuf, "fm_sample", ["mean", "sem"])
)

# %%
opts_cr = {
    "color": "black",
    "ylabel": "fluorescence (A. U.)",
    "xlabel": "time (sec)",
    "aspect": 1.2,
}
opts_lay = {"sublabel_format": None, "fig_size": 80}
opts_vl = {"linestyle": ":", "color": "black"}
sum_sthres["time"] = (sum_sthres.index - 150) * 2 / 30
sum_shuf["time"] = (sum_shuf.index - 150) * 2 / 30
hvres = (
    hv.Spread(sum_sthres, "time", ["mean", "sem"])
    * hv.Curve(sum_sthres, "time", "mean").opts(title="Aligned", **opts_cr)
    * hv.VLine(0).opts(**opts_vl)
    + hv.Spread(sum_shuf, "time", ["mean", "sem"])
    * hv.Curve(sum_shuf, "time", "mean").opts(title="Shuffled", **opts_cr)
    * hv.VLine(0).opts(**opts_vl)
).opts(**opts_lay)

# %%
hvres

# %%
hv.save(hvres, "sthres.svg", backend="matplotlib")

# %%
dpath = "./data/ts44-3/s10/"
param_load_videos = {
    "pattern": r"msCam[0-9]+\.avi$",
    "dtype": np.uint8,
    "downsample": dict(frame=2, height=2, width=2),
    "downsample_strategy": "subset",
}

# %%
minian_ds = open_minian(dpath, backend="zarr")
varr = load_videos(dpath, **param_load_videos)

# %%
# %%time
max_proj = varr.max("frame").compute()
As = minian_ds["A"].sum("unit_id").compute()

# %%
opts_im = {
    "aspect": max_proj.sizes["width"] / max_proj.sizes["height"],
    "cmap": "viridis",
    "xaxis": None,
    "yaxis": None,
}
hvres = (
    hv.Image(max_proj, ["width", "height"]).opts(title="Max projection", **opts_im)
    + hv.Image(As, ["width", "height"]).opts(title="Spatial footprints", **opts_im)
).opts(sublabel_format=None, tight=True)
hvres

# %%
hv.save(hvres, "./max_proj.svg", dpi=500)
