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
            def compute_welch():
                a = fft()
                return (np.abs(a)**2).mean("t")
            welch = await checkpoint_xarray(compute_welch, f"welch/{session_str}.zarr", f"welch", session_index)
            def compute_coh():
                a = fft().to_dataset()
                a["chan_num"] = a["chan_num"].compute()
                df = rawchan_metadata[["chan_group", "structure", "chan_num"]]
                meta = df.to_xarray().rename(index="channel").set_coords(["chan_group", "structure", "chan_num"]).drop_vars("channel")
                meta = meta.set_index(channel="chan_num")
                meta = meta.reindex(channel=a["chan_num"])
                a = xr.merge([a, meta], join="left")
                a = a["data"]
                def select(a1, a2):
                    return (a1["sig_type"] == "eeg" ) | ((a1["sig_type"] == a2["sig_type"]) & (a1["structure"] != a2["structure"]))
                a1, a2 = replicate_dim(a, "channel", 2, select, "stacked", stacked_dim="channel_pair")
                res = (a1*np.conj(a2)).mean("t")
                return res
            coh = await checkpoint_xarray(compute_coh, f"coh/{session_str}.zarr", f"coh", session_index)
            return welch, coh, rawchan_metadata
        # welchs = {}
        # cohs = {}

        # async def process_sig(func, n):
        #     sig = await checkpoint_xarray(func, f"{n}/{session_str}.zarr", n, session_index)
        #     fft = await checkpoint_xarray(lambda: compute_scaled_fft(sig().sel(t=slice(start_t, end_t))).sel(f=slice(None, 120)), f"fft_{n}/{session_str}.zarr", f"fft_{n}", session_index)
        #     psd = await checkpoint_xarray(lambda: compute_psd_from_scaled_fft(fft()), f"psd_{n}/{session_str}.zarr", f"psd_{n}", session_index)
        #     welch = await checkpoint_xarray(lambda: compute_welch_from_psd(psd()), f"welch_{n}/{session_str}.zarr", f"welch_{n}", session_index)
        #     coh = await checkpoint_xarray(lambda: compute_coh_from_psd(psd()), f"coh_{n}/{session_str}.zarr", f"coh_{n}", session_index)

        #     welchs[n] = welch
        #     cohs[n] = coh
        
        # async with anyio.create_task_group() as tg:
        #     for n, func in initial_sigs.items():
        #         tg.start_soon(process_sig, func, n)
            
        # logger.warning("Concatenating")
        # if len(welchs) > 0:
        #     async with lock:
        #         if not do_stop:
        #             do_stop = True
        #             try:
        #                 # print(welchs)
        #                 ar = (welchs["lfp"]()).to_dataset()
        #             # print("\n\n\n\nPRINT\n\n\n\n\n")
        #                 meta = rawchan_metadata.set_index("chan_num").to_xarray().rename(chan_num="channel")
        #                 print(meta)
        #                 print(ar)
        #                 print(xr.merge([ar, meta[["structure", "chan_group"]]], join="inner"))
        #                 # print(ar)
        #             except Exception as e:
        #                 print("PROBLEM")
        #                 print(e)
        #                 raise
                    # finally:
                    #     raise KeyboardInterrupt
        # meta = rawchan_metadata.set_index("chan_num").to_xarray().rename(chan_num="channel")
        # if len(welchs) > 0:
        #     def compute_modify(k, a):
        #         ar: xr.DataArray = a()
        #         ar = ar.to_dataset()
        #         ar = ar.assign_coords(sig_type=k, chan_num=ar["channel"]).drop_vars("channel")
        #         return ar
        #     session_welch = await checkpoint_xarray(lambda: xr.concat([compute_modify(k, a)for k, a in welchs.items()], dim="channel"),
        #                                             f"session_welch/{session_str}.zarr", "session_welch", session_index)
        # if len(cohs) > 0:
        #     session_coh = await checkpoint_xarray(lambda: xr.concat([compute_modify(k, a) for k, a in cohs.items()], dim="channel"),
        #                                             f"session_coh/{session_str}.zarr", "session_coh", session_index)
        # return session_welch, session_coh, rawchan_metadata   
        except Exception as e:
            logger.exception("Error processing session")
            return e
    # session_coh = xr.concat(cohs, dim="channel").assign_coords(session=session_str)
    # return session_welch, session_coh
    
async def process_sessions(analysis_ds: xr.Dataset):
    welchs = []
    cohs = []
    chan_metadatas = []
    errors = []
    async def add_session_result(session_ds, session_index):
            result =  await process_rat_session(session_ds, session_index)
            if isinstance(result, Exception):
                errors.append(result)
            elif result:
                welch, coh, chan_metadata = result
                welchs.append(welch)
                cohs.append(coh)
                chan_metadatas.append(chan_metadata)
            

    
    async with anyio.create_task_group() as tg:
        for s in range(analysis_ds.sizes["session"]):
            ds = analysis_ds.isel(session=s)
            tg.start_soon(add_session_result, ds, s)
            await anyio.sleep(0)
    print(f"#sessions with errors {len(errors)}.")
    for er in errors:
        print(er)
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