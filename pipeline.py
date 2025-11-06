from json import load
import anyio.to_thread
# from dafn.dask_helper import checkpoint_xra_zarr, checkpoint_xrds_zarr, checkpoint_delayed
from dafn.xarray_utilities import xr_merge, chunk, replicate_dim
from pipeline_fct.rats import get_matfile_metadata, get_smrx_metadata, rat_raw_chan_classification, get_mat_df, get_initial_sigs, may_match
from dafn.signal_processing import compute_bua, compute_lfp, compute_scaled_fft
from dafn.spike2 import smrxadc2electrophy, smrxchanneldata, sonfile
from dafn.signal_processing import compute_lfp, compute_bua
from dafn.utilities import flexible_merge, ValidationError
from pipeline_helper import checkpoint_xarray, checkpoint_json, checkpoint_excel, single

from pathlib import Path
import pandas as pd, numpy as np, xarray as xr
import tqdm.auto as tqdm
import re, shutil
import dask, dask.array as da
import anyio
import logging
import h5py
import beautifullogger



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
base_result_path = Path("/media/julienb/T7 Shield/Revue-FRM/AnalysisData2/")

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

error_groups = {}
class AccepTableError(Exception): 
    def __init__(self, group):
        super().__init__(group)
        if not group in error_groups:
            error_groups[group] = tqdm.tqdm(desc=f"Error {group}")
        error_groups[group].update(1)


def get_channel_metadata(spike2_file, mat_file, mat_basefolder):
    spike2_chans = smrxchanneldata(spike2_file)
    mat_chans = get_mat_df(mat_file, mat_basefolder)
    spike2_chans = spike2_chans.loc[spike2_chans["chan_type"].isin(["Adc", "EventRise"])]
    try:
        all_chans = flexible_merge(spike2_chans, mat_chans, "callable", lambda l, r: may_match(l, r, mat_basefolder, spike2_file), validation="1:1!", how="inner", suffixes=("", "_mat"))
    except ValidationError:
        return pd.DataFrame()
    all_chans["chan_group"] = all_chans["chan_name"].str.lower().apply(rat_raw_chan_classification)
    all_chans["chan_group"] = np.where(all_chans["physical_channel"]<0, "neuron", all_chans["chan_group"])
    return all_chans

def compute_initial_sigs(all_chans, spike2_file):
    initial_sigs = get_initial_sigs(all_chans, spike2_file)
    return initial_sigs

async def process_rat_session(session_ds: xr.Dataset, session_index):
    from functools import partial
    async with single(False):
        try:
            session_str = session_ds["session"].item()
            spike2_file = session_ds["spike2_file"].item()
            mat_file = session_ds["mat_file"]

            

            def compute_ffts(initial_sigs):
                fft = compute_scaled_fft(initial_sigs).sel(f=slice(None, 120))
                return fft

            def compute_pwelch(ffts):
                pwelch = (np.abs(ffts)**2).mean("t")
                return pwelch
            
            def compute_coh(ffts, all_chans):
                all_chans_xr = all_chans.to_xarray()[["structure", "chan_num"]].set_coords(["chan_num", "structure"]).drop_vars("index")
                all_chans_xr["structure"] = all_chans_xr["structure"].astype(str)
                fft_merged = xr_merge(ffts.to_dataset(name="fft"), all_chans_xr, on="chan_num", how="left")["fft"]
                def select(a1, a2):
                    return ((a1["sig_type"] == "eeg") | (a2["sig_type"] == "eeg")) | ((a1["sig_type"] == a2["sig_type"]))

                a1, a2 = replicate_dim(fft_merged, "channel", 2, select, "stacked", stacked_dim="channel_pair")
                coh =chunk((a1*np.conj(a2)).mean("t"), channel_pair=1).reset_coords(["structure_1", "structure_2"], drop=True)
                return coh

            all_chans = (await checkpoint_excel(partial(get_channel_metadata, spike2_file, mat_file, mat_basefolder), 
                                f"chan_metadata/{session_str}.xlsx", "chan_metadata", session_index, mode="process"))()
            if all_chans.empty:
                raise AccepTableError("No relevant data")
            if all_chans.loc[all_chans["signal_type"]=="raw"].empty:
                raise AccepTableError("No raw data")
            
            initial_sigs = (await checkpoint_xarray(partial(compute_initial_sigs, all_chans, spike2_file), 
                            f"init_sig/{session_str}.zarr", "init_sig", session_index, mode="process"))
            ffts =  (await checkpoint_xarray(lambda: compute_ffts(initial_sigs()), f"fft/{session_str}.zarr", "fft", session_index))
            pwelch = (await checkpoint_xarray(lambda: compute_pwelch(ffts()), f"pwelch/{session_str}.zarr", "pwelch", session_index))
            coh = (await checkpoint_xarray(lambda: compute_coh(ffts(), all_chans), f"coh/{session_str}.zarr", "coh", session_index))
            return pwelch, coh, all_chans
        except AccepTableError as e:
            return None
        except Exception as e:
            logger.exception(f"Error processing session with index {session_index}")
            raise
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

if __name__ == "__main__":
    anyio.run(main, backend="trio")