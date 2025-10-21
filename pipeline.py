import anyio.to_thread
# from dafn.dask_helper import checkpoint_xra_zarr, checkpoint_xrds_zarr, checkpoint_delayed
from pipeline_fct.rats import get_matfile_metadata, get_smrx_metadata, get_initial_signals, rat_raw_chan_classification
from dafn.signal_processing import compute_psd_from_scaled_fft, compute_coh_from_psd, compute_welch_from_psd
from dafn.signal_processing import compute_bua, compute_lfp, compute_scaled_fft
from dafn.spike2 import smrxadc2electrophy, smrxchanneldata
from dafn.signal_processing import compute_lfp, compute_bua
from pathlib import Path
import pandas as pd, numpy as np, xarray as xr
import tqdm.auto as tqdm
import re, shutil
import dask, dask.array as da
import anyio
import logging
# import beautifullogger
from pipeline_helper import checkpoint_xarray

logger = logging.getLogger(__name__)
# beautifullogger.setup()

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
    analysis_ds["mat_file"] = xr.where(analysis_ds["mat_file"].notnull(), str(mat_basefolder) + "/"+ analysis_ds["mat_file"].astype(str), np.nan)
    return analysis_ds


async def process_rat_session(session_ds, session_index):
    session_str = session_ds["session"].item()
    spike2_file = session_ds["spike2_file"].item()
    all_chans = smrxchanneldata(spike2_file)
    all_raw_chans = all_chans.loc[all_chans["physical_channel"]>= 0].copy()
    all_raw_chans["chan_group"] = all_raw_chans["chan_name"].str.lower().apply(rat_raw_chan_classification)
    probe_chan_nums = all_raw_chans["chan_num"].loc[all_raw_chans["chan_group"]=="Probe"].tolist()
    ipsieeg_chan_nums = all_raw_chans["chan_num"].loc[all_raw_chans["chan_group"]=="EEGIpsi"].tolist()
    initial_sigs = {}
    if len(probe_chan_nums) > 0:
         initial_sigs["lfp"] = lambda: compute_lfp(smrxadc2electrophy(spike2_file, probe_chan_nums))
         initial_sigs["bua"] = lambda: compute_bua(smrxadc2electrophy(spike2_file, probe_chan_nums))
    if len(ipsieeg_chan_nums) == 1:
        initial_sigs["eeg"] = lambda: compute_lfp(smrxadc2electrophy(spike2_file, ipsieeg_chan_nums))

    welchs = {}
    cohs = {}

    async def process_sig(func, n):
        sig = await checkpoint_xarray(func, f"{n}/{session_str}.zarr", n, session_index)
        fft = await checkpoint_xarray(lambda: compute_scaled_fft(sig()).sel(f=slice(None, 120)), f"fft_{n}/{session_str}.zarr", f"fft_{n}", session_index)
        psd = await checkpoint_xarray(lambda: compute_psd_from_scaled_fft(fft()), f"psd_{n}/{session_str}.zarr", f"psd_{n}", session_index)
        welch = await checkpoint_xarray(lambda: compute_welch_from_psd(psd()), f"welch_{n}/{session_str}.zarr", f"welch_{n}", session_index)
        coh = await checkpoint_xarray(lambda: compute_coh_from_psd(psd()), f"coh_{n}/{session_str}.zarr", f"coh_{n}", session_index)

        welchs[n] = welch
        cohs[n] = coh
    
    async with anyio.create_task_group() as tg:
        for n, func in initial_sigs.items():
            tg.start_soon(process_sig, func, n)
        
    # logger.warning("Concatenating")
    if len(welchs) > 0:
        session_welch = await checkpoint_xarray(lambda: xr.concat([a().assign_coords(sig_type=k) for k, a in welchs.items()], dim="channel"),
                                                f"session_welch/{session_str}.zarr", "session_welch", session_index)
    if len(cohs) > 0:
        session_coh = await checkpoint_xarray(lambda: xr.concat([a().assign_coords(sig_type=k) for k, a in cohs.items()], dim="channel"),
                                                f"session_coh/{session_str}.zarr", "session_coh", session_index)
    # session_coh = xr.concat(cohs, dim="channel").assign_coords(session=session_str)
    # return session_welch, session_coh
    
async def process_sessions(analysis_ds: xr.Dataset):
    welchs = []
    cohs = []
    # session_progress = tqdm.tqdm(total=analysis_ds.sizes["session"], desc="Sessions")
    # session_limiter = anyio.CapacityLimiter(5)
    async def add_session_result(session_ds, session_index):
        # async with session_limiter:
            # logger.info("processing session")
            result =  await process_rat_session(session_ds, session_index)
            if result:
                welch, coh = result
                welchs.append(welch)
                cohs.append(coh)
            # session_progress.update()
            # logger.info("session done")

    
    async with anyio.create_task_group() as tg:
        for s in range(analysis_ds.sizes["session"]):
            ds = analysis_ds.isel(session=s)
            tg.start_soon(add_session_result, ds, s)
            # await anyio.sleep(1)

    # all_welch = await checkpoint_xra_zarr(xr.concat(welchs, dim="channel"), base_result_path/f"all_welch.zarr", client)
    # all_coh = await checkpoint_xra_zarr(xr.concat(cohs, dim="channel"), base_result_path/f"all_coh.zarr", client)
    # return all_welch, all_coh

async def main():
    def write(x: xr.Dataset, p: Path):
        x.to_zarr(p)
    analysis_ds: xr.Dataset = (await checkpoint_xarray(
        get_session_info, 
        "analysis_files.zarr", 
        "files", 0
    ))()
    
    # analysis_ds: xr.Dataset = await checkpoint_delayed(
    #     dask.delayed(get_session_info)(), 
    #     base_result_path/"analysis_files.zarr", 
    #     client, write, lambda p: xr.open_zarr(p).compute()
    # )
    analysis_ds = analysis_ds.sortby("session")
    logger.info("got analysis ds")
    print(analysis_ds)
    await process_sessions(analysis_ds)

anyio.run(main, backend="trio")