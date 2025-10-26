import anyio.to_thread
# from dafn.dask_helper import checkpoint_xra_zarr, checkpoint_xrds_zarr, checkpoint_delayed
from pipeline_fct.rats import get_matfile_metadata, get_smrx_metadata, rat_raw_chan_classification, get_mat_df, mat_spike2_raw_join, extract_timing
from dafn.signal_processing import compute_psd_from_scaled_fft, compute_coh_from_psd, compute_welch_from_psd
from dafn.signal_processing import compute_bua, compute_lfp, compute_scaled_fft
from dafn.spike2 import smrxadc2electrophy, smrxchanneldata
from dafn.signal_processing import compute_lfp, compute_bua
from dafn.xarray_utilities import replicate_dim
from pathlib import Path
import pandas as pd, numpy as np, xarray as xr
import tqdm.auto as tqdm
import re, shutil
import dask, dask.array as da
import anyio
import logging
import h5py
import beautifullogger
from pipeline_helper import checkpoint_xarray, checkpoint_json, checkpoint_excel, single

logger = logging.getLogger(__name__)
beautifullogger.setup(displayLevel=logging.WARNING)

import warnings

# Suppress warnings with a matching message regex
warnings.filterwarnings("ignore", message=".*Increasing number of chunks by factor of.*")

# client = dask.distributed.Client(processes=False,  dashboard_address=":8787")
# print("Dashboard running at:", client.dashboard_link)
client= None

spike2files_basefolder = Path("/media/julienb/T7 Shield/Revue-FRM/RawData/Rats/Version-RAW-2-Nico/")
mat_basefolder = Path("/media/julienb/T7 Shield/Revue-FRM/RawData/Rats/Version-Matlab-Nico/")
base_result_path = Path("/media/julienb/T7 Shield/Revue-FRM/AnalysisData/")

def get_session_info():
    spike2files = list(spike2files_basefolder.glob("**/*.smr"))
    spike2df = get_smrx_metadata([p.relative_to(spike2files_basefolder) for p in spike2files])
    mat_files = list(mat_basefolder.glob("**/*.mat"))
    mat_df = get_matfile_metadata([p.relative_to(mat_basefolder) for p in mat_files])
    merged = pd.merge(spike2df, mat_df, on=["condition", "subject", "session", "segment_index"], how="outer")
    core = merged[[col for col in merged.columns if not col in ["mat_file", "structure", "signal_type"]]].drop_duplicates().set_index("spike2_file").to_xarray()
    other = merged[["mat_file", "structure", "signal_type", "spike2_file"]].set_index(["structure", "signal_type", "spike2_file"]).to_xarray()
    analysis_ds = xr.merge([core, other]).rename(session="session_grp").rename(spike2_file="session")
    analysis_ds["spike2_file"] = analysis_ds["session"]
    analysis_ds["session"] = xr.apply_ufunc(lambda p: str(Path(p).with_suffix("")).replace("/", "--"), analysis_ds["spike2_file"], vectorize=True)
    analysis_ds["spike2_file"] = str(spike2files_basefolder) + "/"+ analysis_ds["spike2_file"].astype(str)
    analysis_ds["mat_file"] = xr.where(analysis_ds["mat_file"].notnull(), analysis_ds["mat_file"].astype(str), np.nan)
    return analysis_ds


async def process_rat_session(session_ds, session_index):
    async with single(False):
        try:
            session_str = session_ds["session"].item()
            spike2_file = session_ds["spike2_file"].item()

            def get_rawchan_metadata():
                all_chans = smrxchanneldata(spike2_file)
                rawchan_metadata = all_chans.loc[all_chans["physical_channel"]>= 0].copy()
                mat_df = get_mat_df(session_ds["mat_file"], mat_basefolder)
                rawchan_metadata = rawchan_metadata.merge(mat_spike2_raw_join(mat_df, rawchan_metadata), on=rawchan_metadata.columns.tolist(), how="left")
                return rawchan_metadata
            
            rawchan_metadata = (await checkpoint_excel(get_rawchan_metadata, f"chan_metadata/{session_str}.xlsx", "chan_metadata", session_index))()
            rawchan_metadata["chan_group"] = rawchan_metadata["chan_name"].str.lower().apply(rat_raw_chan_classification)
            probe_chan_nums = rawchan_metadata["chan_num"].loc[(rawchan_metadata["chan_group"]=="Probe") & ~pd.isna(rawchan_metadata["structure"])].tolist()
            ipsieeg_chan_nums = rawchan_metadata["chan_num"].loc[rawchan_metadata["chan_group"]=="EEGIpsi"].tolist()

            if len(probe_chan_nums) == 0:
                return None
            start_t, end_t = (await checkpoint_json(lambda: extract_timing(rawchan_metadata, spike2_file, mat_basefolder),
                                                f"timing/{session_str}.json", "timing", session_index))()
            
            
            def compute_initial_sigs():
                initial_sigs = {}
                if len(probe_chan_nums) > 0:
                    initial_sigs["lfp"] = compute_lfp(smrxadc2electrophy(spike2_file, probe_chan_nums)).assign_coords(sig_type="lfp")
                    initial_sigs["bua"] = compute_bua(smrxadc2electrophy(spike2_file, probe_chan_nums)).assign_coords(sig_type="bua")
                if len(ipsieeg_chan_nums) == 1:
                    initial_sigs["eeg"] = compute_lfp(smrxadc2electrophy(spike2_file, ipsieeg_chan_nums)).assign_coords(sig_type="eeg")
                res = xr.concat([a for a in initial_sigs.values()], dim="channel").sel(t=slice(start_t, end_t))
                return res

            sig = await checkpoint_xarray(compute_initial_sigs, f"init_sig/{session_str}.zarr", "init_sig", session_index)
            fft = await checkpoint_xarray(lambda: compute_scaled_fft(sig()).sel(f=slice(None, 120)), 
                                        f"fft/{session_str}.zarr", f"fft", session_index)
            
            def add_metadata_to_array(a: xr.DataArray):
                a = a.to_dataset(name="data")
                a["chan_num"] = a["chan_num"].compute()
                df = rawchan_metadata[["chan_group", "structure", "chan_num"]]
                meta = df.to_xarray().rename(index="channel").set_coords(["chan_group", "structure", "chan_num"]).drop_vars("channel")
                meta = meta.set_index(channel="chan_num")
                meta = meta.reindex(channel=a["chan_num"])
                a = xr.merge([a, meta], join="left")
                a = a.drop_vars("channel")
                return a["data"]
            def compute_welch():
                a = fft()
                a = add_metadata_to_array(a)
                return (np.abs(a)**2).mean("t")
            
            def compute_coh():
                a = fft()
                a = add_metadata_to_array(a)
                def select(a1, a2):
                    return ((a1["sig_type"] == "eeg") & (a2["sig_type"] != "eeg")) | ((a1["sig_type"] == a2["sig_type"]) & (a1["structure"] != a2["structure"]))
                a1, a2 = replicate_dim(a, "channel", 2, select, "stacked", stacked_dim="channel_pair")
                res = (a1*np.conj(a2)).mean("t")
                return res
            welch = await checkpoint_xarray(compute_welch, f"welch/{session_str}.zarr", f"welch", session_index)
            coh = await checkpoint_xarray(compute_coh, f"coh/{session_str}.zarr", f"coh", session_index)
            return welch, coh, rawchan_metadata  
        except Exception as e:
            logger.exception("Error processing session")
            return e
    
async def process_sessions(analysis_ds: xr.Dataset):
    analysis_ds = analysis_ds
    # .isel(session=slice(0, 180))
    results = {}
    errors = []
    async def add_session_result(session_ds, session_index):
            result =  await process_rat_session(session_ds, session_index)
            if isinstance(result, Exception):
                errors.append(result)
            elif result:
                results[session_index] = result
            
    async with anyio.create_task_group() as tg:
        for s in range(analysis_ds.sizes["session"]):
            ds = analysis_ds.isel(session=s)
            tg.start_soon(add_session_result, ds, s)
            await anyio.sleep(0)
    print(f"#sessions with errors {len(errors)}.")
    for er in errors:
        print(er)
    print(f"Got {len(results)} valid results")
    session_metadata_cols = ["condition", "has_swa", "is_APO", "session_grp", "subject", "session"]

    def combine_welch():
        welchs = []
        for k, (w, _, meta) in results.items():
            ar: xr.DataArray = w().compute()
            session_ds = analysis_ds.isel(session=k)
            session_info = {k:session_ds[k].item() for k in session_metadata_cols}
            ar = ar.assign_coords(session_info)
            welchs.append(ar)
        res = xr.concat(welchs, dim="channel").compute()
        return res
    
    def combine_coherence():
        coherence = []
        for k, (_, coh, meta) in tqdm.tqdm(results.items()):
            ar: xr.DataArray = coh().compute()
            session_ds = analysis_ds.isel(session=k)
            session_info = {k:session_ds[k].item() for k in session_metadata_cols}
            ar = ar.assign_coords(session_info)
            coherence.append(ar)
        res = xr.concat(coherence, dim="channel_pair").compute()
        print("Storing to zarr")
        return res
    
    all_welch = (await checkpoint_xarray(combine_welch, f"all_welch.zarr", f"all_welch", 0))()
    all_coh = (await checkpoint_xarray(combine_coherence, f"all_coh.zarr", f"all_coh", 0))()

    

    # print(welchs[0]().compute())
    # print(chan_metadatas[0])
    # all_welch = await checkpoint_xra_zarr(xr.concat(welchs, dim="channel"), base_result_path/f"all_welch.zarr", client)
    # all_coh = await checkpoint_xra_zarr(xr.concat(cohs, dim="channel"), base_result_path/f"all_coh.zarr", client)
    # return all_welch, all_coh

async def main():
    analysis_ds: xr.Dataset = (await checkpoint_xarray(
        get_session_info, 
        "analysis_files.zarr", 
        "files", 0
    ))()
    analysis_ds = analysis_ds.sortby("session")
    logger.info("got analysis ds")
    print(analysis_ds)
    await process_sessions(analysis_ds)

anyio.run(main, backend="trio")