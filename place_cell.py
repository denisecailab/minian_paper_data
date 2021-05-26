# %%
import os
import sys
import numpy as np
import xarray as xr
import pandas as pd
import holoviews as hv
import dask as da
from dask.distributed import Client, LocalCluster
from sklearn.mixture import GaussianMixture
from scipy.ndimage import label, gaussian_filter1d
from scipy.signal import medfilt

# MINIAN_PATH = "./minian_snapshot"
# sys.path.append(MINIAN_PATH)

# from minian.utilities import open_minian
from minian_snapshot.minian.utilities import open_minian, xrconcat_recursive

# %%
def map_ts(ts: pd.DataFrame) -> pd.DataFrame:
    """map frames from Cam1 to Cam0 with nearest neighbour using the timestamp file from miniscope recordings.
    
    Parameters
    ----------
    ts : pd.DataFrame
        input timestamp dataframe. should contain field 'frameNum', 'camNum' and 'sysClock'
    
    Returns
    -------
    pd.DataFrame
        output dataframe. should contain field 'fmCam0' and 'fmCam1'
    """
    ts_sort = ts.sort_values("sysClock")
    ts_sort["ts_behav"] = np.where(ts_sort["camNum"] == 1, ts_sort["sysClock"], np.nan)
    ts_sort["ts_forward"] = ts_sort["ts_behav"].fillna(method="ffill")
    ts_sort["ts_backward"] = ts_sort["ts_behav"].fillna(method="bfill")
    ts_sort["diff_forward"] = np.absolute(ts_sort["sysClock"] - ts_sort["ts_forward"])
    ts_sort["diff_backward"] = np.absolute(ts_sort["sysClock"] - ts_sort["ts_backward"])
    ts_sort["fm_behav"] = np.where(ts_sort["camNum"] == 1, ts_sort["frameNum"], np.nan)
    ts_sort["fm_forward"] = ts_sort["fm_behav"].fillna(method="ffill")
    ts_sort["fm_backward"] = ts_sort["fm_behav"].fillna(method="bfill")
    ts_sort["fmCam1"] = np.where(
        ts_sort["diff_forward"] < ts_sort["diff_backward"],
        ts_sort["fm_forward"],
        ts_sort["fm_backward"],
    )
    ts_map = (
        ts_sort[ts_sort["camNum"] == 0][["frameNum", "fmCam1"]]
        .dropna()
        .rename(columns=dict(frameNum="fmCam0"))
        .astype(dict(fmCam1=int))
    )
    ts_map["fmCam0"] = ts_map["fmCam0"] - 1
    ts_map["fmCam1"] = ts_map["fmCam1"] - 1
    return ts_map


def process_behav(
    behav: pd.DataFrame, run_dim="X", wnd=31, thres_dx=0.05, thres_rw=20,
) -> pd.DataFrame:
    """differntiate locations based on running directions,
    and filter out frames when the animal is not moving.
    The output locaion will have the same sign as running speed.
    
    Parameters
    ----------
    behav : pd.DataFrame
        input dataframe of behavior tracking results, should contain a column with name `run_dim`.
    run_dim : str, optional
        the dimension along which the animal is running, by default "X"
    wnd : int, optional
        the window size in frames where the running speed is estimated and media-filtered, by default 31
    thres_dx : float, optional
        the threshold for change of pixel per frame along `run_dim`, below which all the frames will be discarded, by default 0.01
    
    Returns
    -------
    pd.DataFrame
        output dataframe
    """
    behav["dx"] = medfilt(np.gradient(behav[run_dim], wnd), wnd)
    behav["stationary"] = thres_gmm(np.abs(behav["dx"]).values, com=0)
    xmax, xmin = behav["X"].max() - thres_rw, behav["X"].min() + thres_rw
    behav["reward_zone"] = ~behav["X"].between(xmin, xmax)
    rw_low = (behav["X"] < xmin).astype(int)
    rw_high = (behav["X"] > xmax).astype(int)
    trans_low = [(t, "low") for t in np.where(rw_low.diff() == -1)[0]]
    trans_high = [(t, "high") for t in np.where(rw_high.diff() == -1)[0]]
    trans = (
        pd.DataFrame(trans_low + trans_high, columns=("index", "reward"))
        .sort_values("index")
        .reset_index(drop=True)
    )
    if trans.iloc[0]["reward"] == "low":
        trans["r"] = trans["reward"].map({"low": 1, "high": 0})
    else:
        trans["r"] = trans["reward"].map({"low": 0, "high": 1})
    trans = trans[trans["r"].diff().fillna(1) == 1]
    behav["trial"] = 0
    behav.loc[trans["index"], "trial"] = 1
    behav["trial"] = behav["trial"].cumsum().astype(int)
    dx_sign = np.sign(behav["dx"])
    behav[run_dim] = behav[run_dim] * dx_sign
    behav = behav[(~behav["stationary"]) & (~behav["reward_zone"])]
    return behav


def norm(a: np.ndarray) -> np.ndarray:
    """normalize input array to the range of [0, 1]. Can handle Nan and zero range.
    
    Parameters
    ----------
    a : np.ndarray
        input array.
    
    Returns
    -------
    np.ndarray
        normalized array.
    """
    amin = np.nanmin(a)
    return (a - amin) / (np.nanmax(a) - amin + np.finfo(float).eps)


def gaussian_nan(a, **kwargs):
    nan_mask = np.isnan(a)
    v = np.nan_to_num(a)
    w = np.where(nan_mask, 0, 1).astype(np.float)
    ag = gaussian_filter1d(v, **kwargs) / gaussian_filter1d(w, **kwargs)
    return np.where(nan_mask, np.nan, ag)


def compute_fr(
    S: xr.DataArray, bin_dim="x", nbins=100, normalize=True, sigma=2.5
) -> xr.DataArray:
    """compute averaged firing rate by binning along the 'frame' dimension according to `bin_dim`.
    
    Parameters
    ----------
    S : xr.DataArray
        input data representing raw spikes. presumably the S matrix from CNMF
    bin_dim : str, optional
        the dimension according to which the spikes are binned and averaged, by default 'x'
    nbins : int, optional
        number of bins, by default 100
    normalize : bool, optional
        whether to normalize result, by default True
    
    Returns
    -------
    xr.DataArray
        output firing rate
    """
    bdim = bin_dim + "_bins"
    fr = S.groupby_bins(bin_dim, nbins).mean(dim="frame")
    fr = fr.assign_coords({bdim: np.arange(fr.sizes[bdim])}).rename("fr")
    if normalize:
        fr = xr.apply_ufunc(
            norm,
            fr.chunk({bdim: -1}),
            input_core_dims=[[bdim]],
            output_core_dims=[[bdim]],
            vectorize=True,
            dask="parallelized",
            output_dtypes=[fr.dtype],
        )
    if sigma is not None:
        fr = xr.apply_ufunc(
            gaussian_nan,
            fr.chunk({bdim: -1}),
            input_core_dims=[[bdim]],
            output_core_dims=[[bdim]],
            vectorize=True,
            kwargs={"sigma": sigma},
            dask="parallelized",
            output_dtypes=[fr.dtype],
        )
    return fr


def compute_occp(S: xr.DataArray, bin_dim="x", nbins=100, sigma=2.5) -> xr.DataArray:
    """calculate the occupancy based on count of frames in each bin according to `bin_dim`.
    
    Parameters
    ----------
    S : xr.DataArray
        input data representing raw spikes. presumably the S matrix from CNMF
    bin_dim : str, optional
        the dimension according to which frames are counted, by default 'x'
    nbins : int, optional
        number of bins, by default 100
    
    Returns
    -------
    xr.DataArray
        output occupancy array
    """
    bdim = bin_dim + "_bins"
    occp = S[bin_dim].groupby_bins(bin_dim, nbins).count() / S[bin_dim].count()
    occp = occp.assign_coords({bdim: np.arange(occp.sizes[bdim])}).rename("occp")
    if sigma is not None:
        occp = xr.apply_ufunc(
            gaussian_nan,
            occp.chunk({bdim: -1}),
            input_core_dims=[[bdim]],
            output_core_dims=[[bdim]],
            vectorize=True,
            dask="parallelized",
            kwargs={"sigma": sigma},
            output_dtypes=[occp.dtype],
        )
    return occp


def compute_si(fr: xr.DataArray, occp: xr.DataArray, agg_dim="x_bins") -> xr.DataArray:
    """compute spatial information using binned firing rates and occupancy
    
    Parameters
    ----------
    fr : xr.DataArray
        input firing rates
    occp : xr.DataArray
        input occupancy
    agg_dim : str, optional
        the dimension along which to aggreagate, by default "x_bins"
    
    Returns
    -------
    xr.DataArray
        output spatial information
    """
    mfr = fr.mean(agg_dim)
    return (occp * fr / mfr * np.log2(fr / mfr, where=fr != 0)).sum(agg_dim)


def compute_stb(S: xr.DataArray, **kwargs) -> xr.DataArray:
    tmax = np.max(S.coords["trial"])
    fr_odd = compute_fr(S.sel(frame=(S.coords["trial"] % 2 == 1)), **kwargs)
    fr_even = compute_fr(S.sel(frame=(S.coords["trial"] % 2 == 0)), **kwargs)
    fr_first = compute_fr(S.sel(frame=(S.coords["trial"] < tmax / 2)), **kwargs)
    fr_last = compute_fr(S.sel(frame=(S.coords["trial"] > tmax / 2)), **kwargs)
    z_oe = compute_corr(fr_odd, fr_even)
    z_fl = compute_corr(fr_first, fr_last)
    return (z_fl + z_oe) / 2


def compute_corr(fr1: xr.DataArray, fr2: xr.DataArray) -> xr.DataArray:
    m1, m2 = fr1.mean("x_bins"), fr2.mean("x_bins")
    s1, s2 = fr1.std("x_bins"), fr2.std("x_bins")
    r = ((fr1 - m1) * (fr2 - m2)).mean("x_bins") / (s1 * s2)
    return np.arctanh(r)


def thres_gmm(a: xr.DataArray, com=-1) -> xr.DataArray:
    """binnarize input array using gaussian mixture model.
    
    Parameters
    ----------
    a : xr.DataArray
        input array
    
    Returns
    -------
    xr.DataArray
        binnarized array
    """
    a = a.reshape(-1, 1)
    gmm = GaussianMixture(n_components=2)
    gmm.fit(a)
    idg = np.argsort(gmm.means_.reshape(-1))[com]
    return (gmm.predict(a) == idg).reshape(-1)


def thres_psize(fr: xr.DataArray, qthres: float, sz_thres: int) -> bool:
    """return whether a cell is place cell based on the place field size criteria:
    the place field is defined as the longest continuous region where the averaged
    firing rate in that region exceeds `qthres` percentile of all firing rates,
    and then the place field must be larger than `sz_thres` spatial bins for a cell
    to be classified as place cell.
    
    Parameters
    ----------
    fr : xr.DataArray
        input firing rate of a cell
    qthres : float
        quantile threshold to define place field
    sz_thres : int
        threshold for size of place field
    
    Returns
    -------
    bool
        whether a cell is a place cell
    """
    q = np.nanquantile(fr, qthres)
    lab, nlab = label(fr > q)
    if nlab:
        len_ls = [np.sum(lab == lb + 1) for lb in range(nlab)]
        if max(len_ls) > sz_thres:
            max_lb = np.argmax(len_ls)
            com = np.mean(np.where(lab == max_lb + 1)[0])
            return com
        else:
            return np.nan


def process_place(
    dpath: str, nbins=200, nshuf=1000, sig_thres=0.95, sz_qthres=0.95, sz_thres=2
) -> xr.Dataset:
    try:
        ts = pd.read_csv(os.path.join(dpath, "timestamp.dat"), delimiter="\t")
        behav = (
            pd.read_csv(os.path.join(dpath, "behavConcat_LocationOutput.csv"))[
                ["X", "Y", "Distance_px"]
            ]
            .reset_index(drop=True)
            .rename_axis("fmCam1")
            .reset_index()
        )
        minian_ds = open_minian(dpath, backend="zarr")
        print("processing {}".format(dpath))
    except:
        print("file missing under {}".format(dpath))
        return xr.Dataset()
    S = minian_ds["S"].chunk({"frame": -1})
    behav = process_behav(behav)
    fmap = map_ts(ts)
    fmap = fmap.merge(behav, how="left", on="fmCam1").set_index("fmCam0")
    try:
        S = S.assign_coords(
            x=("frame", fmap["X"][S.coords["frame"].values]),
            trial=("frame", fmap["trial"][S.coords["frame"].values]),
        )
    except KeyError:
        print("behavior mapping error, check timestamp file")
        return xr.Dataset()
    # S_thres = (
    #     xr.apply_ufunc(
    #         thres_gmm,
    #         S,
    #         input_core_dims=[["frame"]],
    #         output_core_dims=[["frame"]],
    #         vectorize=True,
    #         dask="parallelized",
    #         output_dtypes=[bool],
    #     )
    #     .rename("S_thres")
    #     .persist()
    # )
    S_thres = S
    fr = compute_fr(S_thres, nbins=nbins).compute().rename("fr")
    stb = compute_stb(S_thres, nbins=nbins).compute().rename("stb")
    occp = compute_occp(S_thres, nbins=nbins).compute().rename("occp")
    si = compute_si(fr, occp).compute().rename("si")
    sh_ls = []
    for sh in np.random.random_integers(0, S_thres.sizes["frame"], nshuf):
        S_sh = S_thres.roll(frame=sh, roll_coords=False)
        sh_ls.append(S_sh)
    S_shuf = xr.concat(sh_ls, "shuf").chunk({"shuf": "auto"})
    fr_shuf = compute_fr(S_shuf, nbins=nbins).compute()
    si_shuf = compute_si(fr_shuf, occp).compute()
    stb_shuf = compute_stb(S_shuf, nbins=nbins).compute()
    mask_si = (si > si_shuf.quantile(sig_thres, "shuf")).rename("mask_si")
    mask_stb = (stb > stb_shuf.quantile(sig_thres, "shuf")).rename("mask_stb")
    maxpos = xr.apply_ufunc(
        thres_psize,
        fr,
        input_core_dims=[["x_bins"]],
        output_core_dims=[[]],
        vectorize=True,
        kwargs={"qthres": sz_qthres, "sz_thres": sz_thres},
    ).rename("maxpos")
    return xr.merge([S_thres, fr, occp, si, mask_si, stb, mask_stb, maxpos])


def vec_corr(fr0: xr.DataArray, fr1: xr.DataArray, agg_dim="x_bins", vec_dim="index"):
    mask = fr0.notnull().all(vec_dim) & fr1.notnull().all(vec_dim)
    fr0, fr1 = fr0.where(mask), fr1.where(mask)
    fr0_mean = fr0.mean(agg_dim)
    fr1_mean = fr1.mean(agg_dim)
    fr0_var = ((fr0 - fr0_mean) ** 2).sum(agg_dim)
    fr1_var = ((fr1 - fr1_mean) ** 2).sum(agg_dim)
    return ((fr0 - fr0_mean) * (fr1 - fr1_mean)).sum(agg_dim) / np.sqrt(
        fr0_var * fr1_var
    )


# %%
if __name__ == "__main__":
    dpath = "./data/pfd2"
    cluster = LocalCluster(dashboard_address="0.0.0.0:9999")
    client = Client(cluster)
    ds_ls = []
    for root, dirs, files in os.walk(dpath):
        if root.count(os.path.sep) > (dpath.count(os.path.sep) + 2):
            continue
        ds = process_place(root, nshuf=1000)
        ds_ls.append(ds)
    plc_ds = xrconcat_recursive(list(filter(bool, ds_ls)), ["animal", "session"])
    print(plc_ds)
    plc_ds.to_netcdf("./data/inter/place_cells.nc")
