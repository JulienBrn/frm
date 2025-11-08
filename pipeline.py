from json import load
from os import error
import anyio.to_thread
from requests import session
# from dafn.dask_helper import checkpoint_xra_zarr, checkpoint_xrds_zarr, checkpoint_delayed
from dafn.xarray_utilities import xr_merge, chunk, replicate_dim
from pipeline_fct.rats import get_matfile_metadata, get_smrx_metadata, rat_raw_chan_classification, get_mat_df, get_initial_sigs, may_match
from dafn.signal_processing import compute_bua, compute_lfp, compute_scaled_fft
from dafn.spike2 import smrxadc2electrophy, smrxchanneldata, sonfile
from dafn.signal_processing import compute_lfp, compute_bua
from dafn.utilities import flexible_merge, ValidationError
from pipeline_helper import checkpoint_xarray, checkpoint_json, checkpoint_excel, single, add_note_to_exception, set_limiter, set_base_result_path

from pathlib import Path
import pandas as pd, numpy as np, xarray as xr
import tqdm.auto as tqdm
import re, shutil
import dask, dask.array as da
import anyio
import logging
import h5py
import beautifullogger
from typing import Tuple

print = lambda a: tqdm.tqdm.write(str(a))
xr.set_options(display_max_rows=100)

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
base_result_path = Path("/media/julienb/T7 Shield/Revue-FRM/AnalysisData3/")

set_base_result_path(base_result_path)
set_limiter(5)
error_groups = {}
class AccepTableError(Exception): 
    def __init__(self, group):
        super().__init__(group)
        if not group in error_groups:
            error_groups[group] = tqdm.tqdm(desc=f"Error {group}", leave=False)
        error_groups[group].update(1)
        error_groups[group].refresh()



def get_session_info():
    spike2files = [p for p in spike2files_basefolder.glob("**/*.smr") if not "Events files" in str(p)]
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
    analysis_ds["subject"] = analysis_ds["subject"].astype(str)
    return analysis_ds


def get_channel_metadata(spike2_file, mat_file, mat_basefolder):
    spike2_chans = smrxchanneldata(spike2_file)
    mat_chans = get_mat_df(mat_file, mat_basefolder)
    spike2_chans = spike2_chans.loc[spike2_chans["chan_type"].isin(["Adc", "EventRise"])]
    try:
        if mat_chans.empty:
            raise AccepTableError("No mat data")
        if spike2_chans.loc[spike2_chans["physical_channel"]>=0].empty:
            raise AccepTableError("No raw data in spike2 file")
        try:
            all_chans = flexible_merge(spike2_chans, mat_chans, "callable", lambda l, r: may_match(l, r, mat_basefolder, spike2_file), validation="1:1!", how="inner", suffixes=("", "_mat"))
        except ValidationError:
            raise AccepTableError("Not 1:1! merge...")
        all_chans["chan_group"] = all_chans["chan_name"].str.lower().apply(rat_raw_chan_classification)
        all_chans["chan_group"] = np.where(all_chans["physical_channel"]<0, "neuron", all_chans["chan_group"])
        if all_chans.loc[all_chans["signal_type"]=="raw"].empty:
            raise AccepTableError("No raw data after merge")
    except AccepTableError as e:
        error_df = pd.DataFrame()
        error_df["__error__"] = [e.args[0]]
        return error_df
    return all_chans


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

            all_chans: pd.DataFrame = (await checkpoint_excel(partial(get_channel_metadata, spike2_file, mat_file, mat_basefolder), 
                                f"chan_metadata/{session_str}.xlsx", "chan_metadata", session_index, mode="process"))()
            
            if "__error__" in all_chans.columns:
                raise AccepTableError(all_chans["__error__"].iat[0])
            
            all_chans["chan_name_normalized"] = all_chans["chan_name"].str.translate(str.maketrans('', '', "_-/ "))
            if all_chans["chan_name_normalized"].duplicated().any():
                raise Exception("Not unique normalized names...")
            initial_sigs = (await checkpoint_xarray(partial(get_initial_sigs, all_chans, spike2_file), 
                            f"init_sig/{session_str}.zarr", "init_sig", session_index, mode="process"))
            ffts =  (await checkpoint_xarray(lambda: compute_ffts(initial_sigs()), f"fft/{session_str}.zarr", "fft", session_index))
            pwelch = (await checkpoint_xarray(lambda: compute_pwelch(ffts()), f"pwelch/{session_str}.zarr", "pwelch", session_index))
            coh = (await checkpoint_xarray(lambda: compute_coh(ffts(), all_chans), f"coh/{session_str}.zarr", "coh", session_index))
            return pwelch, coh, all_chans
        except AccepTableError as e:
            return None
        except Exception as e:
            add_note_to_exception(e, f"While processing session with index {session_index}")
            raise
    

def compute_neuron_type(meta, eeg, delta_t, duration, spike2_file):
    from pipeline_fct.rats import get_probe_data
    import scipy.signal
    eeg_sig = get_probe_data(eeg, delta_t, duration, spike2_file).isel(channel=0).compute()
    bp_params = scipy.signal.butter(4, [0.2, 2], btype="bandpass", output="sos", fs=eeg_sig["t"].attrs["fs"])
    eeg_sig = xr.apply_ufunc(lambda s: scipy.signal.sosfiltfilt(bp_params, s), eeg_sig, input_core_dims=[["t"]], output_core_dims=[["t"]])
    neuron_types = []
    with sonfile(spike2_file) as rec:
        time_base = rec.GetTimeBase()
        for _, row in meta.loc[(meta["chan_group"] == "neuron") & (meta["structure"] == "GPe")].iterrows():
            chan = int(row["chan_num"])
            spike_times = np.array(rec.ReadEvents(chan, 10**7, int(delta_t/time_base), int((delta_t+duration)/time_base)))*time_base
            values = eeg_sig.interp(t=spike_times)
            proportion = (values.sum()/np.abs(values).sum()).item()
            neuron_types.append(dict(chan_num=chan, proportion=proportion))
    return pd.DataFrame(neuron_types)

async def process_neuron_types(session_ds, meta: pd.DataFrame, session_index):
    from functools import partial
    try:
        spike2_file = session_ds["spike2_file"].item()
        session_str = session_ds["session"].item()

        meta = meta.loc[meta["chan_group"].isin(["EEGIpsi", "neuron"])]
        delta_t = meta["delta_t"].max()
        duration = meta["common_duration"].min()
        eeg = meta.loc[meta["chan_group"]=="EEGIpsi"]
        if eeg.empty:
            raise AccepTableError("No eeg data in swa session")
        elif len(eeg.index) > 1:
            raise Exception("More than one eeg signal...")
        
        res =  (await checkpoint_excel(partial(compute_neuron_type, meta, eeg, delta_t, duration, spike2_file), 
                                f"neuron_type/{session_str}.xlsx", "neuron_type", session_index, mode="process"))()
        if res.empty:
            raise AccepTableError("No GPe neurons in swa session")
        res = pd.merge(res, meta[["chan_num", "chan_name_normalized", "mat_key"]], how="left", on="chan_num")
        return res[["proportion", "chan_name_normalized"]]
    except AccepTableError as e:
        return None
    except Exception as e:
        add_note_to_exception(e, f"While processing neuron types on session with index {session_index}")
        raise

async def get_rats_combined_data(analysis_ds: xr.Dataset) -> Tuple[xr.DataArray, xr.DataArray, xr.Dataset, xr.Dataset]:
    analysis_ds = analysis_ds
    results = {}
    neuron_types = {}

    async def add_session_result(session_ds, session_index):
        result =  await process_rat_session(session_ds, session_index)
        if result:
            results[session_index] = result

    async def add_neuron_types(session_ds, meta, session_index):
        grp = (session_ds["subject"].item(), session_ds["session_grp"].item())
        if not grp in neuron_types:
            neuron_types[grp] = session_index
        else:
            raise Exception(f"Multiple swa session for group {grp}, got {neuron_types[grp]} {session_index}")
        result =  await process_neuron_types(session_ds, meta, session_index)
        if result is not None:
            neuron_types[grp] = result
        else:
            del neuron_types[grp]
            
    async with anyio.create_task_group() as tg:
        for s in range(analysis_ds.sizes["session"]):
            ds = analysis_ds.isel(session=s)
            tg.start_soon(add_session_result, ds, s)
            await anyio.sleep(0)

    async with anyio.create_task_group() as tg:
        for k, (_, _, meta) in results.items():
            session_ds = analysis_ds.isel(session=k)
            if session_ds["has_swa"] == 1:
                tg.start_soon(add_neuron_types, session_ds, meta, k)
                await anyio.sleep(0)


    def combine_welch():
        all = []
        for k, (a, _, meta) in results.items():
            ar: xr.DataArray = a().compute().assign_coords(session_index=k)
            all.append(ar)
        res = xr.concat(all, dim="channel")
        return res
    
    def combine_coherence():
        all = []
        for k, (_, a, meta) in results.items():
            ar: xr.DataArray = a().compute().assign_coords(session_index=k)
            all.append(ar)
        res = xr.concat(all, dim="channel_pair")
        return res
    
    def combine_meta():
        all = []
        for k, (_, _, meta) in results.items():
            ar: xr.Dataset = meta.to_xarray()
            ar = ar.drop_vars(["index"]+[c for c in ar.data_vars if "unnamed" in c.lower()])
            ar = ar.set_coords(ar.data_vars).assign_coords(session_index=k)
            all.append(ar)
        res = xr.concat(all, dim="index")
        for v in res.coords:
            if res[v].dtype==object:
                res[v] = res[v].astype(str)
        return res
    
    def combine_neuron_types():
        all = []
        for (subject, sgrp), nt in neuron_types.items():
            ar: xr.Dataset = nt.to_xarray()
            ar = ar.drop_vars(["index"])
            ar = ar.set_coords(ar.data_vars).assign_coords(subject=subject, session_grp=sgrp)
            all.append(ar)
        res = xr.concat(all, dim="index")
        for v in res.coords:
            if res[v].dtype==object:
                res[v] = res[v].astype(str)
        print(res)
        return res

    
    all_welch = (await checkpoint_xarray(combine_welch, f"combined_welch.zarr", f"combined_welch", 0))()
    all_coh = (await checkpoint_xarray(combine_coherence, f"combined_coh.zarr", f"combined_coh", 0))()
    all_meta = (await checkpoint_xarray(combine_meta, f"combined_meta.zarr", f"combined_meta", 0))()
    all_nt = (await checkpoint_xarray(combine_neuron_types, f"combined_nt.zarr", f"combined_nt", 0))()

    return all_welch, all_coh, all_meta, all_nt


async def process_rats_data(analysis_ds):
    all_welch, all_coh, all_meta, neuron_types = await get_rats_combined_data(analysis_ds)
    neuron_types["neuron_type"] = xr.where(neuron_types["proportion"] > 0.1, "Arky", xr.where(neuron_types["proportion"] < -0.1, "Proto", "")).astype(str)
    neuron_types=neuron_types.set_coords("neuron_type")
    print(neuron_types)
    channel_metadata = xr_merge(all_meta, analysis_ds.drop_dims(["signal_type", "structure"]), how="left", on=["session_index"])
    channel_metadata = xr_merge(channel_metadata, neuron_types, how="left", on=["session_grp", "subject", "chan_name_normalized"])
    channel_metadata["has_swa"] = channel_metadata["has_swa"].astype(bool)
    channel_metadata["neuron_type"] = channel_metadata["neuron_type"].fillna("").astype(str)
    channel_df = channel_metadata.to_dataframe().reset_index(drop=True)
    channel_df.astype({col: "int" for col in channel_df.select_dtypes("bool").columns}).to_excel(base_result_path/"rat_metadata.xlsx", index=False)
    print(channel_df.columns)
    print(channel_df[["is_APO", "has_swa"]])

    session_columns = ["subject", "session_index", "session", "condition", "has_swa", "is_APO", "session_grp"]
    channel_columns = ["structure", "neuron_type", "chan_name", "chan_num"]
    channel_metadata = channel_metadata.reset_coords()
    all_welch = xr_merge(all_welch.to_dataset(name="pwelch").reset_coords(), channel_metadata[channel_columns + session_columns], 
                         how="left", on=["session_index", "chan_num"])
    all_coh = all_coh.to_dataset(name="coh").reset_coords()
    for suffix in ["_1", "_2"]:
        all_coh = xr_merge(all_coh, channel_metadata[session_columns+channel_columns].rename({k:k+suffix for k in channel_columns}).set_coords([k for k in session_columns if not k=="session_index"]), 
                           how="left", on=["session_index", "chan_num"+suffix])
    all_welch = all_welch.set_coords([c for c in all_welch.data_vars if not c=="pwelch"])["pwelch"]
    all_coh = all_coh.set_coords([c for c in all_coh.data_vars if not c=="coh"])["coh"]
    for k in session_columns:
        all_welch[k] = all_welch[k].broadcast_like(all_welch["channel"])
        all_coh[k] = all_coh[k].broadcast_like(all_coh["channel_pair"])
    return all_welch, all_coh

async def main():
    analysis_ds: xr.Dataset = (await checkpoint_xarray(get_session_info, "analysis_files.zarr",  "files", 0))()
    analysis_ds = analysis_ds.sortby("session").assign_coords(session_index=("session", np.arange(analysis_ds.sizes["session"]))).set_coords(analysis_ds.data_vars)
    logger.info("got analysis ds")
    for v in analysis_ds.coords:
        if analysis_ds[v].dtype==object:
            analysis_ds[v] = analysis_ds[v].astype(str)
    print(analysis_ds)
    rat_welch, rat_coh = await process_rats_data(analysis_ds)
    from pipeline_helper import _save_xarray
    _save_xarray(rat_welch, base_result_path/"rat_welch.zarr")
    _save_xarray(rat_coh, base_result_path/"rat_coh.zarr")
    print(rat_welch)
    print(rat_coh)

if __name__ == "__main__":
    anyio.run(main, backend="trio")