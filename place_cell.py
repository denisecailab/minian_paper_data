import os
import sys
import numpy as np
import xarray as xr
import pandas as pd
import holoviews as hv
from sklearn.mixture import GaussianMixture
from scipy.ndimage import label
from scipy.signal import medfilt

# MINIAN_PATH = "./minian_snapshot"
# sys.path.append(MINIAN_PATH)

# from minian.utilities import open_minian
from minian_snapshot.minian.utilities import open_minian


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
    ts["change_point"] = ts["camNum"].diff()
    ts["ts_behav"] = np.where(ts["change_point"] == 1, ts["sysClock"], np.nan)
    ts["ts_forward"] = ts["ts_behav"].fillna(method="ffill")
    ts["ts_backward"] = ts["ts_behav"].fillna(method="bfill")
    ts["diff_forward"] = np.absolute(ts["sysClock"] - ts["ts_forward"])
    ts["diff_backward"] = np.absolute(ts["sysClock"] - ts["ts_backward"])
    ts["fm_behav"] = np.where(ts["change_point"] == 1, ts["frameNum"], np.nan)
    ts["fm_forward"] = ts["fm_behav"].fillna(method="ffill")
    ts["fm_backward"] = ts["fm_behav"].fillna(method="bfill")
    ts["fmCam1"] = np.where(
        ts["diff_forward"] < ts["diff_backward"], ts["fm_forward"], ts["fm_backward"]
    )
    ts_map = (
        ts[ts["camNum"] == 0][["frameNum", "fmCam1"]]
        .dropna()
        .rename(columns=dict(frameNum="fmCam0"))
        .astype(dict(fmCam1=int))
    )
    ts_map["fmCam0"] = ts_map["fmCam0"] - 1
    ts_map["fmCam1"] = ts_map["fmCam1"] - 1
    return ts_map


def run_direction(
    behav: pd.DataFrame, run_dim="X", wnd=31, thres_dx=0.01
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
    behav = behav[behav["dx"].abs() > thres_dx]
    behav[run_dim] = behav[run_dim] * np.sign(behav["dx"])
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
    return (a - np.nanmin(a)) / (np.nanmax(a) - np.nanmin(a) + np.finfo(float).eps)


def compute_fr(S: xr.DataArray, bin_dim="x", nbins=100, normalize=True) -> xr.DataArray:
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
    return fr


def compute_occp(S: xr.DataArray, bin_dim="x", nbins=100) -> xr.DataArray:
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
    mfr = fr.mean(agg_dim).compute()
    return (occp * fr * np.log2(fr / mfr + np.finfo(np.float32).eps)).sum(agg_dim)


def thres_gmm(a: xr.DataArray) -> xr.DataArray:
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
    idg = np.argsort(gmm.means_.reshape(-1))[-1]
    return (gmm.predict(a) == idg).reshape(-1).astype(int)


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
    dpath: str, nbins=200, nshuf=1000, si_qthres=0.95, sz_qthres=0.95, sz_thres=2
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
    behav = run_direction(behav)
    fmap = map_ts(ts)
    fmap = fmap.merge(behav, on="fmCam1").set_index("fmCam0")
    try:
        S = S.assign_coords(x=("frame", fmap["X"][S.coords["frame"].values]))
    except KeyError:
        print("behavior mapping error, check timestamp file")
        return xr.Dataset()
    S_thres = (
        xr.apply_ufunc(
            thres_gmm,
            S,
            input_core_dims=[["frame"]],
            output_core_dims=[["frame"]],
            vectorize=True,
            dask="parallelized",
            output_dtypes=[int],
        )
        .rename("S_thres")
        .compute()
    )
    fr = compute_fr(S_thres, nbins=nbins).compute().rename("fr")
    occp = compute_occp(S_thres, nbins=nbins).compute().rename("occp")
    si = compute_si(fr, occp).compute().rename("si")
    shuf_ls = []
    for sh in np.random.random_integers(0, S_thres.sizes["frame"], nshuf):
        S_sh = S_thres.roll(frame=sh, roll_coords=False)
        fr_sh = compute_fr(S_sh, nbins=nbins)
        si_sh = compute_si(fr_sh, occp)
        shuf_ls.append(si_sh)
    si_shuf = xr.concat(shuf_ls, "shuf").compute()
    mask_si = (si > si_shuf.quantile(si_qthres, "shuf")).rename("mask_si")
    maxpos = xr.apply_ufunc(
        thres_psize,
        fr,
        input_core_dims=[["x_bins"]],
        output_core_dims=[[]],
        vectorize=True,
        kwargs={"qthres": sz_qthres, "sz_thres": sz_thres},
    ).rename("maxpos")
    return xr.merge([S_thres, fr, occp, si, mask_si, maxpos])


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


if __name__ == "__main__":
    process_place("./data/ts45-4/s11")
