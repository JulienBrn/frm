from json import load
from os import error
import anyio.to_thread
from requests import session
# from dafn.dask_helper import checkpoint_xra_zarr, checkpoint_xrds_zarr, checkpoint_delayed
from dafn.xarray_utilities import xr_merge, chunk, replicate_dim
from pipeline_fct.rats import get_matfile_metadata, get_smrx_metadata, rat_raw_chan_classification, get_mat_df, get_initial_sigs, may_match
from dafn.signal_processing import compute_bua, compute_lfp, compute_scaled_fft, compute_coherence, spiketimes_to_continuous
from dafn.spike2 import smrxadc2electrophy, smrxchanneldata, sonfile
from dafn.signal_processing import compute_lfp, compute_bua
from dafn.utilities import flexible_merge, ValidationError
from pipeline_helper import checkpoint_xarray, checkpoint_json, checkpoint_excel, single, add_note_to_exception, set_limiter, set_base_result_path, ErrorReport

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
from unionfind import unionfind

import pipeline_helper
old_print = print
print = lambda a: tqdm.tqdm.write(str(a))
xr.set_options(display_max_rows=100)

logger = logging.getLogger(__name__)
beautifullogger.setup(displayLevel=logging.WARNING)

import warnings

# Suppress warnings with a matching message regex
warnings.filterwarnings("ignore", message=".*Increasing number of chunks by factor of.*")

client= None

spike2files_basefolder = Path("/media/julienb/T7 Shield/Revue-FRM/RawData/Rats/Version-RAW-2-Nico/")
mat_basefolder = Path("/media/julienb/T7 Shield/Revue-FRM/RawData/Rats/Version-Matlab-Nico/")
monkey_basefolder = Path("/media/julienb/T7 Shield/Revue-FRM/RawData/Monkeys/")
human_basefolder = Path("/media/julienb/T7 Shield/Revue-FRM/RawData/Humans/")
base_result_path = Path("/media/julienb/T7 Shield/Revue-FRM/AnalysisData8/")

set_base_result_path(base_result_path)
set_limiter(5)


if __name__ == "__main__": #To protect against multiprocessing
    error_report = ErrorReport(base_result_path/"reported_errors.tsv")



def get_session_info():
    spike2files = [p for p in spike2files_basefolder.glob("**/*.smr") if not "event" in str(p).lower()]
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
    
    spike2_chans["chan_group"] = spike2_chans["chan_name"].str.lower().apply(rat_raw_chan_classification)
    spike2_chans["chan_group"] = np.where(spike2_chans["physical_channel"]<0, "neuron", spike2_chans["chan_group"])
    all_chans = flexible_merge(spike2_chans, mat_chans, "callable", lambda l, r: may_match(l, r, mat_basefolder, spike2_file), how="outer", suffixes=("", "_mat"))
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
                coh =chunk(compute_coherence(a1, a2), channel_pair=1).reset_coords(["structure_1", "structure_2"], drop=True)
                return coh

            all_chans: pd.DataFrame = (await checkpoint_excel(partial(get_channel_metadata, spike2_file, mat_file, mat_basefolder), 
                                f"chan_metadata/{session_str}.xlsx", "chan_metadata", session_index, mode="process"))()
            if all_chans.empty:
                error_report.report_error("No Data", spike2_file)
            all_chans = all_chans.loc[all_chans["chan_type"].isin(["Adc", "EventRise"])]
            if all_chans.empty:
                error_report.report_error("No Data after chantype filtering", spike2_file)
            if all_chans.loc[all_chans["chan_group"] == "EEGIpsi"].empty:
                error_report.report_error("No EEG for session", spike2_file, f"chan_metadata/{session_str}.xlsx", raise_excpt=False)
            if not "mat_key" in all_chans:
                error_report.report_error("No matched mat channel", spike2_file)
            all_chans = all_chans.loc[~all_chans["mat_key"].isna() | (all_chans["chan_group"]=="EEGIpsi")]
            if all_chans["chan_num"].isna().any():
                # print(all_chans)
                error_report.report_error("Mat chan not found in spike2", spike2_file, f"chan_metadata/{session_str}.xlsx", raise_excpt=False)
            all_chans = all_chans.loc[~all_chans["chan_num"].isna()]
                # print(all_chans)
                # raise Exception("Stop")
            if all_chans.empty:
                error_report.report_error("Empty channels after merge", spike2_file, f"chan_metadata/{session_str}.xlsx")
            if all_chans.duplicated("chan_num").any():
                error_report.report_error("Duplicated spike2 channels", spike2_file, f"chan_metadata/{session_str}.xlsx", raise_excpt=False)
            if all_chans.duplicated("mat_key").any():
                error_report.report_error("Duplicated spike2 channels", spike2_file, f"chan_metadata/{session_str}.xlsx", raise_excpt=False)
            all_chans=all_chans.loc[~(all_chans.duplicated("chan_num", keep=False) | all_chans.duplicated("mat_key", keep=False))]
            all_chans["chan_num"] = all_chans["chan_num"].astype(int)
            # all_chans["is_APO"] = all_chans["is_APO"].astype(bool)
            # all_chans["has_swa"] = all_chans["has_swa"].astype(bool)
            

            if all_chans.loc[all_chans["chan_group"] == "EEGIpsi"].empty:
                error_report.report_error("No EEG for session after merge", spike2_file, False)
            if all_chans.loc[all_chans["physical_channel"] >= 0].empty:
                error_report.report_error("No raw data for session after merge", spike2_file)
            if all_chans.loc[all_chans["chan_group"] != "EEGIpsi"].empty:
                error_report.report_error("Only eeg data after merge", spike2_file)
            
            all_chans["delta_t"] =  all_chans["delta_t"].ffill().bfill()
            all_chans["common_duration"] =  all_chans["common_duration"].ffill().bfill()

                
            all_chans["chan_name_normalized"] = all_chans["chan_name"].str.translate(str.maketrans('', '', "_-/ "))
            if all_chans["chan_name_normalized"].duplicated().any():
                error_report.report_error("Not unique normalized names...", spike2_file)
            initial_sigs = (await checkpoint_xarray(partial(get_initial_sigs, all_chans, spike2_file), 
                            f"init_sig/{session_str}.zarr", "init_sig", session_index, mode="process"))
            ffts =  (await checkpoint_xarray(lambda: compute_ffts(initial_sigs()), f"fft/{session_str}.zarr", "fft", session_index))
            pwelch = (await checkpoint_xarray(lambda: compute_pwelch(ffts()), f"pwelch/{session_str}.zarr", "pwelch", session_index))
            coh = (await checkpoint_xarray(lambda: compute_coh(ffts(), all_chans), f"coh/{session_str}.zarr", "coh", session_index))
            def compute_n_spikes(sig_type, spike2file, chan, delta_t, common_duration):
                if sig_type != "units":
                    return np.nan
                with sonfile(spike2_file) as rec:
                    time_base = rec.GetTimeBase()
                    data = np.array(rec.ReadEvents(chan, 10**7, int(delta_t/time_base), int((delta_t+common_duration)/time_base)))*time_base
                    return data.size/common_duration
            
            all_chans["neuron_fr"] = all_chans.apply(lambda row: compute_n_spikes(row["signal_type"], spike2_file, row["chan_num"], row["delta_t"], row["common_duration"]), axis=1)
            all_chans["neuron_str_type"] = np.where((all_chans["structure"]=="STR") & (all_chans["signal_type"] =="units") & all_chans["neuron_fr"].notna(),
                                                    np.where(all_chans["neuron_fr"] < 2, "STR<2Hz", "STR>2Hz"), "")
            return pwelch, coh, all_chans
        except ErrorReport.ReportedError as e:
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
            error_report.report_error("No eeg data in swa session", spike2_file)
        elif len(eeg.index) > 1:
            raise Exception("More than one eeg signal...")
        
        res =  (await checkpoint_excel(partial(compute_neuron_type, meta, eeg, delta_t, duration, spike2_file), 
                                f"neuron_type/{session_str}.xlsx", "neuron_type", session_index, mode="process"))()
        if res.empty:
            error_report.report_error("No GPe neurons in swa session", spike2_file)
        res = pd.merge(res, meta[["chan_num", "chan_name_normalized", "mat_key"]], how="left", on="chan_num")
        return res[["proportion", "chan_name_normalized"]]
    except ErrorReport.ReportedError as e:
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
            tg.start_soon(add_session_result, ds, ds["session_index"].item())
            await anyio.sleep(0)

    async with anyio.create_task_group() as tg:
        for k, (_, _, meta) in results.items():
            session_ds = analysis_ds.isel(session=k)
            if session_ds["has_swa"] == 1:
                tg.start_soon(add_neuron_types, session_ds, meta, session_ds["session_index"].item())
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
    
    def combine_meta() -> xr.Dataset:
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
    
    def combine_neuron_types() -> xr.Dataset:
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
        return res
    
    

    all_welch = (await checkpoint_xarray(combine_welch, f"combined_welch.zarr", f"combined_welch", 0))()
    all_coh = (await checkpoint_xarray(combine_coherence, f"combined_coh.zarr", f"combined_coh", 0))()
    all_meta: xr.Dataset = (await checkpoint_xarray(combine_meta, f"combined_meta.zarr", f"combined_meta", 0))()
    all_nt = (await checkpoint_xarray(combine_neuron_types, f"combined_nt.zarr", f"combined_nt", 0))()
    return all_welch, all_coh, all_meta, all_nt


async def process_rats_data():
    analysis_ds: xr.Dataset = (await checkpoint_xarray(get_session_info, "analysis_files.zarr",  "files", 0))()
    analysis_ds = analysis_ds.sel(session=(analysis_ds["mat_file"]!="").any(["signal_type", "structure"]))
    if analysis_ds[["session"]].to_dataframe().reset_index()["session"].duplicated().any():
        raise Exception("Not unique sessions")
    # print(analysis_ds)
    # raise Exception("Stop")
    analysis_ds = analysis_ds.sortby("session").assign_coords(session_index=("session", np.arange(analysis_ds.sizes["session"]))).set_coords(analysis_ds.data_vars)
    logger.info("got analysis ds")
    for v in analysis_ds.coords:
        if analysis_ds[v].dtype==object:
            analysis_ds[v] = analysis_ds[v].astype(str)
    all_welch, all_coh, all_meta, neuron_types = await get_rats_combined_data(analysis_ds)
    neuron_types["neuron_type"] = xr.where(neuron_types["proportion"] > 0.1, "Arky", xr.where(neuron_types["proportion"] < -0.1, "Proto", "")).astype(str)
    neuron_types=neuron_types.set_coords("neuron_type")
    # print(neuron_types)
    # print(all_meta.to_dataframe().reset_index()["condition"].unique())
    # print(analysis_ds.drop_dims(["signal_type", "structure"]))
    channel_metadata = xr_merge(all_meta, analysis_ds.drop_dims(["signal_type", "structure"]), how="left", on=["session_index"])
    n_per_grp = channel_metadata.to_dataframe().reset_index().groupby(["is_APO", "subject", "session_grp", "condition"])["has_swa"].nunique()
    for _, row in n_per_grp.loc[n_per_grp!=2].reset_index().iterrows():
        if not row["is_APO"]:
            error_report.report_error("No swa session", row["condition"] + "--" + row["subject"]+"--"+row["session_grp"], raise_excpt=False)
    channel_metadata = xr_merge(channel_metadata, neuron_types, how="left", on=["session_grp", "subject", "chan_name_normalized"])
    channel_metadata["has_swa"] = channel_metadata["has_swa"].astype(bool)
    channel_metadata["neuron_type"] = channel_metadata["neuron_type"].fillna("").astype(str)
    channel_metadata["neuron_type"] = xr.where(channel_metadata["neuron_str_type"] != "", channel_metadata["neuron_str_type"], channel_metadata["neuron_type"])
    channel_df = channel_metadata.to_dataframe().reset_index(drop=True)
    channel_df.astype({col: "int" for col in channel_df.select_dtypes("bool").columns}).to_excel(base_result_path/"rat_metadata.xlsx", index=False)
    # print(channel_df.columns)
    # print(channel_df[["is_APO", "has_swa"]])

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





def compute_monkey_raw_data(grp: pd.DataFrame, fs):
    from pipeline_fct.monkey import make_raw_chan
    raw = make_raw_chan(grp, monkey_basefolder, fs)
    bua = compute_bua(raw)
    ffts = compute_scaled_fft(bua)
    pwelch = (np.abs(ffts)**2).mean("t")
    return pwelch

def compute_monkey_neuron_data(row: pd.Series, fs, neuron_fs=1000):
    def get_continuous_data():
        from scipy.io import loadmat
        data: np.ndarray = loadmat(monkey_basefolder / row["file_path"], variable_names=["SUA"])["SUA"][0]/fs
        n_values = int((row["end"] - row["start"]+2)*neuron_fs)
        t = np.arange(n_values)/neuron_fs
        indices = (data*neuron_fs).astype(int)
        ar = np.zeros(t.size, dtype=float)
        np.add.at(ar, indices, 1.0)
        neuron_continuous = xr.DataArray(ar, dims="t")
        neuron_continuous["t"] = row["start"]+t
        neuron_continuous["t"].attrs["fs"] = neuron_fs
        return neuron_continuous
    raw = get_continuous_data()
    ffts = compute_scaled_fft(raw)
    pwelch = (np.abs(ffts)**2).mean("t")
    return pwelch



async def process_monkey_data():
    from pipeline_fct.monkey import get_electrode_groups
    from functools import partial
    analysis_df = pd.read_csv(monkey_basefolder/"BothMonkData_withTime.csv", header=1)
    analysis_df["structure"] = analysis_df.pop("struct").str.slice(0, 3)
    analysis_df["file_path"] = (
        analysis_df["condition"] + "/" + analysis_df["monkey"] + "/" + analysis_df["structure"] + "/" + analysis_df["session"].astype(str) + "/" 
        + "unit"+analysis_df["unit"].astype(str) + ".mat"
    ) 
    analysis_df["raw_chan"] = analysis_df.groupby(["condition", "monkey","structure", "elec", "session"], group_keys=False).apply(get_electrode_groups, include_groups=False)
    analysis_df["raw_id"] = (
        analysis_df["condition"] + "--" + analysis_df["monkey"] + "--" + analysis_df["structure"] + "--" 
        + analysis_df["session"].astype(str) + "--" + analysis_df["elec"].astype(str) + "--" + analysis_df["raw_chan"].astype(str)
    )
    analysis_df["neuron_id"] = (
        analysis_df["condition"] + "--" + analysis_df["monkey"] + "--" + analysis_df["structure"] + "--" 
        + analysis_df["session"].astype(str) + "--" + analysis_df["elec"].astype(str) + "--" + analysis_df["unit"].astype(str)
    )

    results = {}
    results_neuron = {}
    async def process_raw_monkey(grp, id, priority):
        res = await checkpoint_xarray(partial(compute_monkey_raw_data, grp, 25000), f"monkey/welch/{id}.zarr", "monkey_welch", priority, "process")
        results[id] = res

    async def process_neuron_monkey(row, id, priority):
        res = await checkpoint_xarray(partial(compute_monkey_neuron_data, row, 40000), f"monkey/neuron_welch/{id}.zarr", "monkey_neuron_welch", priority)
        results_neuron[id] = res

    async with anyio.create_task_group() as tg:
        for ind, (id, grp) in enumerate(analysis_df.groupby("raw_id", group_keys=False)):
            tg.start_soon(process_raw_monkey, grp, id, ind)

    async with anyio.create_task_group() as tg:
        for ind, (_, row) in enumerate(analysis_df.iterrows()):
            tg.start_soon(process_neuron_monkey, row, row["neuron_id"], ind)

    def combine_welch() -> xr.DataArray:
        all = []
        for k, a in results.items():
            ar: xr.DataArray = a().compute().assign_coords(raw_id=k, sig_type="bua")
            all.append(ar)
        res = xr.concat(all, dim="channel")
        return res
    
    def combine_neuron_welch() -> xr.DataArray:
        all = []
        for k, a in results_neuron.items():
            ar: xr.DataArray = a().compute().assign_coords(neuron_id=k, sig_type="neuron")
            all.append(ar)
        res = xr.concat(all, dim="channel")
        return res
    
    all_welch = (await checkpoint_xarray(combine_welch, f"monkey/combined_welch.zarr", f"monkey_combined_welch", 0))()
    raw_ds = analysis_df[["condition", "monkey", "session", "elec", "structure", "raw_id"]].drop_duplicates("raw_id").to_xarray().drop_vars("index").rename_dims(index="channel")
    raw_ds = xr_merge(all_welch.to_dataset(name="welch"), raw_ds.set_coords(raw_ds.data_vars), on="raw_id", how="left").sel(f=slice(None, 120))["welch"]


    all_neuron_welch = (await checkpoint_xarray(combine_neuron_welch, f"monkey/combined_neuron_welch.zarr", f"monkey_combined_neuron_welch", 0))()
    neuron_ds = analysis_df[["condition", "monkey", "session", "elec", "structure", "neuron_id"]].to_xarray().drop_vars("index").rename_dims(index="channel")
    neuron_ds["neuron_id"] = neuron_ds["neuron_id"].astype(str)
    neuron_ds = xr_merge(all_neuron_welch.to_dataset(name="welch"), neuron_ds.set_coords(neuron_ds.data_vars), on="neuron_id", how="left").sel(f=slice(None, 120))["welch"]
    # print(neuron_ds)
    ds = xr.concat([raw_ds.rename(raw_id="id"), neuron_ds.rename(neuron_id="id")], dim="channel")

    ds = ds.rename(monkey="subject")
    # print(ds)
    return ds
    

def compute_human_nonstn_raw_welch(path, fs):
    from scipy.io import loadmat
    data = loadmat(human_basefolder/path, squeeze_me=True, variable_names=["MUA"])["MUA"]
    start = 0 
    data = xr.DataArray(data.flatten(), dims="t")
    data["t"] = np.arange(data.sizes["t"])/fs +start
    data["t"].attrs["fs"] = fs
    bua = compute_bua(data)
    ffts = compute_scaled_fft(bua).sel(f=slice(None, 120))
    pwelch = (np.abs(ffts)**2).mean("t")
    return pwelch

def compute_human_nonstn_unit_welch(path):
    from scipy.io import loadmat
    data = loadmat(human_basefolder/path, squeeze_me=True, variable_names=["SUA"])["SUA"]
    data = spiketimes_to_continuous(data)
    ffts = compute_scaled_fft(data).sel(f=slice(None, 120))
    pwelch = (np.abs(ffts)**2).mean("t")
    return pwelch

async def process_nonstn_human_data():
    non_stn_folder = human_basefolder / "HumanData4review"
    def get_files():
        non_stn_files = [str(f.relative_to(human_basefolder)) for f in non_stn_folder.glob("**/unit*.mat")]
        non_stn_df = pd.DataFrame()
        non_stn_df["path"] = non_stn_files
        non_stn_df[["structure", "session", "electrode", "unit"]] = non_stn_df["path"].str.extract("HumanData4review/(.*)/(.*)/(.*)/unit(\d+)\.mat")
        non_stn_df = non_stn_df.loc[~non_stn_df["structure"].str.contains("STN")]
        return non_stn_df
    non_stn_df = (await checkpoint_excel(get_files, f"humans/nonstnfiles.xls", "nonstnfiles", 0))()
    non_stn_df["condition"] = "Park"
    non_stn_df["raw_id"] = non_stn_df["structure"] + "--" + non_stn_df["session"].astype(str) + "--" + non_stn_df["electrode"].astype(str)
    non_stn_df["subject"] = non_stn_df["structure"] + "--" + non_stn_df["session"].astype(str)
    non_stn_df["neuron_id"] = non_stn_df["raw_id"]+"--"+non_stn_df["unit"].astype(str)
    non_stn_raw = (non_stn_df.drop_duplicates("raw_id").drop(columns=["neuron_id", "unit"])).sort_values("raw_id")
    non_stn_df = non_stn_df.drop(columns="raw_id")

    all_nonstn_raw = {}
    async def handle_raw(path, raw_id, session, ind):
        from functools import partial
        year = int(session[:4])
        if year >=2015:
            fs = 48076.92337036133
        else:
            fs = 44000
        res = await checkpoint_xarray(partial(compute_human_nonstn_raw_welch, path, fs), f"humans/raw_pwelch/{raw_id}.xr.zarr", "raw_humans_pwelch", ind)
        all_nonstn_raw[raw_id] = res

    async with anyio.create_task_group() as tg:
        for i, row in non_stn_raw.reset_index(drop=True).iterrows():
            tg.start_soon(handle_raw, row["path"], row["raw_id"], row["session"], i)
        
    all_nonstn_units = {}
    async def handle_units(path, neuron_id, ind):
        from functools import partial
        res = await checkpoint_xarray(partial(compute_human_nonstn_unit_welch, path), f"humans/unit_pwelch/{neuron_id}.xr.zarr", "unit_humans_pwelch", ind)
        all_nonstn_units[neuron_id] = res

    async with anyio.create_task_group() as tg:
        for i, row in non_stn_df.reset_index(drop=True).iterrows():
            tg.start_soon(handle_units, row["path"], row["neuron_id"], i)

    def combine_raw_welch() -> xr.DataArray:
        all = []
        for k, a in all_nonstn_raw.items():
            ar: xr.DataArray = a().compute().assign_coords(raw_id=k, sig_type="bua")
            all.append(ar)
        res = xr.concat(all, dim="channel")
        return res
    
    def combine_neuron_welch() -> xr.DataArray:
        all = []
        for k, a in all_nonstn_units.items():
            ar: xr.DataArray = a().compute().assign_coords(neuron_id=k, sig_type="neuron")
            all.append(ar)
        res = xr.concat(all, dim="channel")
        return res
    print(non_stn_df)
    print(non_stn_df.to_xarray().drop_vars("index").rename_dims(index="channel"))
    all_welch_raw = (await checkpoint_xarray(combine_raw_welch, f"humans/combined_raw_welch.zarr", f"humans_combined_raw_welch", 0))()
    raw_ds = non_stn_raw.to_xarray().drop_vars("index").rename_dims(index="channel")
    raw_ds = xr_merge(all_welch_raw.to_dataset(name="welch"), raw_ds.set_coords(raw_ds.data_vars), on="raw_id", how="left")["welch"]
    all_neuron_welch = (await checkpoint_xarray(combine_neuron_welch, f"humans/combined_neuron_welch.zarr", f"humans_combined_neuron_welch", 0))()
    neuron_ds = non_stn_df.to_xarray().drop_vars("index").rename_dims(index="channel")
    neuron_ds = xr_merge(all_neuron_welch.to_dataset(name="welch"), neuron_ds.set_coords(neuron_ds.data_vars), on="neuron_id", how="left")["welch"]
    ds = xr.concat([raw_ds.rename(raw_id="id").drop_vars("path"), neuron_ds.rename(neuron_id="id").drop_vars(["unit", "path"])], dim="channel")
    return ds

def compute_human_stn_raw_welch(path, start_t, end_t, channel):
    from scipy.io import loadmat
    raw_key, fs_key = f'CElectrode{channel}',f'CElectrode{channel}_KHz_Orig'
    data = loadmat(human_basefolder/path, squeeze_me=True, variable_names=[raw_key, fs_key])
    fs, start = data[fs_key]*1000, 0
    data = xr.DataArray(data[raw_key].flatten(), dims="t")
    data["t"] = np.arange(data.sizes["t"])/fs +start
    data = data.sel(t=slice(start_t, end_t))
    data["t"].attrs["fs"] = fs
    bua = compute_bua(data)
    ffts = compute_scaled_fft(bua).sel(f=slice(None, 120))
    pwelch = (np.abs(ffts)**2).mean("t")
    pwelch.attrs["orig_fs"] = fs
    return pwelch

class tmpExcpt(Exception):
    def __init__(self, msg, unit):
        super().__init__(msg)
        self.unit=unit

def compute_human_stn_unit_welch(path, unit_list, start_t, end_t, fs):
    from scipy.io import loadmat
    mat_data = loadmat(str(human_basefolder/path), squeeze_me=True, mat_dtype=True)
    initial_sorting_results = pd.DataFrame(mat_data["sortingResults"]["classifiedSpikes"].item()[:, :2], columns=["unit", "t"])
    initial_sorting_results["t"] = initial_sorting_results["t"]/fs
    sorting_results = initial_sorting_results.loc[(initial_sorting_results["t"] >= start_t) & (initial_sorting_results["t"] <= end_t)]
    
    arrays = []
    for unit in unit_list:
        spiketimes = (sorting_results.loc[sorting_results["unit"] == unit]["t"]).to_numpy()
        try:
            data = spiketimes_to_continuous(spiketimes, start=start_t, end=end_t).assign_coords(unit=unit)
        except Exception:
            # print(f"{unit}, {start_t}, {end_t}")
            # print(sorting_results)
            # print(initial_sorting_results)
            if spiketimes.size==0:
                raise tmpExcpt("No spiketimes", unit)
            raise
        arrays.append(data)
    data = xr.concat(arrays, dim="channel")
    data["t"].attrs["fs"] = 1000
    ffts = compute_scaled_fft(data).sel(f=slice(None, 120))
    pwelch = (np.abs(ffts)**2).mean("t")
    return pwelch
    
async def process_stn_human_data():
    from functools import partial
    stn_folder = human_basefolder / "Human_STN_Correct_All"
    def get_files():
        from scipy.io import loadmat
        all_stn_metadata = []
        for f in stn_folder.glob("all_Sorting_data*.mat"):
            data = loadmat(f, squeeze_me=True)
            data = data[next(k for k in data.keys() if not k.startswith("__"))]
            df = pd.DataFrame(data)
            df = df.where(df.astype(str) != "[]", np.nan)
            df = df.loc[:, df.notna().any(axis=0)]
            df.columns = ["folder"]+ [k.replace("Isolaton", "isolation").lower() for k in df.iloc[0, 1:].astype(str).tolist()]
            df = df.iloc[1:].reset_index(drop=True)
            df["folder"] = np.where(df["folder"].astype(str).str.match("^\d*$"), np.nan, df["folder"])
            df["folder"] = df["folder"].ffill()
            df = df.loc[df["file"].notna()]
            all_stn_metadata.append(df)
        df = pd.concat(all_stn_metadata)
        df["path"] = str(stn_folder.relative_to(human_basefolder))+"/"+"STN_"+df["folder"].str.replace("\\", "/")+"/"+df["file"].str.replace(".map", ".mat")
        df["unit_path"] = df["path"].apply(lambda p: str(Path(p).parent)) +"/"+("sorting results_" + df["path"].apply(lambda p: Path(p).parent.name) 
                            + "-" + df["path"].apply(lambda p: Path(p).stem)+"-channel"+df["channel"].astype(str) + "-"
                            +  "1.mat"
        )
        df["relfolder"] = df["path"].apply(lambda p: str(Path(p).parent))
        
        stability_db = list(stn_folder.glob("**/stabilityDatabase*.mat"))
        all_stability = []
        for p in stability_db:
            data = loadmat(human_basefolder/p, squeeze_me=True)['stabilityDatabase']
            st_db_df = pd.DataFrame(data[1:])
            st_db_df.columns=data[0]
            st_db_df["relfolder"] = str(p.parent.relative_to(human_basefolder))
            st_db_df["file"] = st_db_df.pop("file name")
            all_stability.append(st_db_df)
        all_stability = pd.concat(all_stability)
        if all_stability.duplicated(["relfolder", "channel", "file"]).any():
            raise Exception("problem")
        all = pd.merge(df, all_stability, on=["relfolder", "channel", "file"], how="left")
        all[["structure", "session"]] = all["folder"].str.split("\\", expand=True)
        all["structure"] = "STN_" + all["structure"]
        all["electrode"] = "E"+all["channel"].astype(str)+"_"+all["path"].apply(lambda p: Path(p).stem)
        return all
    stn_df = (await checkpoint_excel(get_files, f"humans/stnfiles.xls", "stnfiles", 0))()
    stn_df["raw_id"] = stn_df["structure"] + "--" + stn_df["session"].astype(str) + "--" + stn_df["electrode"].astype(str) + stn_df["channel"].astype(str)
    stn_df["start"] = stn_df["start rms stable section (s)"]
    stn_df["end"] = stn_df["end rms stable section (s)"]
    stn_df = stn_df[["raw_id", "structure", "session","electrode", "channel", "path", "unit_path", "start", "end"] + [k for k in stn_df.columns if "unit " in k ]]

    if stn_df["raw_id"].duplicated().any():
        raise Exception("Duplication problem")
    def format_df(stn_df):
        stn_df = stn_df.set_index([k for k in stn_df.columns if not "isolation" in k.lower() and not "rate" in k.lower()])
        stn_df.columns = stn_df.columns.str.split(" ", expand=True)
        stn_df: pd.DataFrame = stn_df["unit"]
        stn_df.columns.names = ["unit", None]

        stn_df = stn_df.stack(level="unit", future_stack=True).reset_index()
        stn_df["isolation"] = stn_df["isolation"].astype(float)
        stn_df = stn_df[stn_df["isolation"].notna()]
        stn_df["condition"] = "Park"
        stn_df["subject"] = stn_df["structure"] + "--" + stn_df["session"].astype(str)
        return stn_df
    stn_df = (await checkpoint_excel(partial(format_df, stn_df), f"humans/stnfilesstacked.xls", "stnfilesstacked", 0))()
    stn_df["neuron_id"] = stn_df["raw_id"]+"--"+stn_df["unit"].astype(str)
    stn_df = stn_df.loc[stn_df["isolation"]>0.6]

    all_stn_raw = {}
    all_stn_units = {}

    async def handle_grp(grp: pd.DataFrame, ind):
        from functools import partial
        raw_id, path, unit_path, channel, start, end = (grp[k].iat[0] for k in ['raw_id', "path", "unit_path", "channel", "start", "end"])
        bua = await checkpoint_xarray(partial(compute_human_stn_raw_welch, path, start, end, channel), f"humans/raw_pwelch_stn/{raw_id}.xr.zarr", "raw_humans_pwelch_stn", ind)
        fs = bua().attrs["orig_fs"]
        try:
            neurons = await checkpoint_xarray(partial(compute_human_stn_unit_welch, unit_path, grp["unit"].tolist(), start, end, fs), f"humans/unit_pwelch_stn/{raw_id}.xr.zarr", "unit_humans_pwelch_stn", ind)
        except tmpExcpt as e:
             error_report.report_error("No spiketimes", path, e.unit, raise_excpt=False)
             return
            #  raise
        all_stn_raw[raw_id] = bua
        all_stn_units[raw_id] = neurons


    async with anyio.create_task_group() as tg:
        for i,(_, grp) in enumerate(stn_df.groupby("raw_id")):
            tg.start_soon(handle_grp, grp, i)
        

    def combine_raw_welch() -> xr.DataArray:
        all = []
        for k, a in all_stn_raw.items():
            ar: xr.DataArray = a().compute().assign_coords(raw_id=k, sig_type="bua")
            all.append(ar)
        res = xr.concat(all, dim="channel")
        return res
    
    def combine_neuron_welch() -> xr.DataArray:
        all = []
        for k, a in all_stn_units.items():
            ar: xr.DataArray = a().compute().assign_coords(raw_id=k, sig_type="neuron")
            all.append(ar)
        res = xr.concat(all, dim="channel")
        return res

    all_welch_raw = (await checkpoint_xarray(combine_raw_welch, f"humans/stn_combined_raw_welch.zarr", f"stn_humans_combined_raw_welch", 0))()
    stn_df["mat_channel"] = stn_df.pop("channel")
    stn_df = stn_df[["structure", "session", "electrode", "unit", "raw_id", "neuron_id", "subject", "condition"]]
    raw_ds = stn_df.drop_duplicates("raw_id").to_xarray().drop_vars("index").rename_dims(index="channel")
    raw_ds = xr_merge(all_welch_raw.to_dataset(name="welch"), raw_ds.set_coords(raw_ds.data_vars), on="raw_id", how="left")["welch"]
    all_neuron_welch = (await checkpoint_xarray(combine_neuron_welch, f"humans/stn_combined_neuron_welch.zarr", f"stn_humans_combined_neuron_welch", 0))()
    neuron_ds = stn_df.to_xarray().drop_vars("index").rename_dims(index="channel")
    neuron_ds = xr_merge(all_neuron_welch.to_dataset(name="welch"), neuron_ds.set_coords(neuron_ds.data_vars), on=["raw_id", "unit"], how="left")["welch"]
    print(neuron_ds)
    print(raw_ds)
    ds = xr.concat([raw_ds.rename(raw_id="id").drop_vars(["neuron_id", "unit"]), neuron_ds.rename(neuron_id="id").drop_vars(["unit", "raw_id"])], dim="channel")
    return ds
    

async def process_human_data():
    non_stn = await process_nonstn_human_data()
    stn = await process_stn_human_data()
    return xr.concat([non_stn, stn], dim="channel")
    # from pipeline_fct.humans import get_human_metadata
    # metadata = (await checkpoint_excel(lambda: get_human_metadata(human_basefolder), f"humans/metadata.xls", "human_metadata", 0))()
    # print(metadata)


async def main():
    from pipeline_helper import _save_xarray
    rat_welch, rat_coh = await process_rats_data()
    # rat_welch["neuron_type"] = xr.where((rat_welch["structure"] == "STR") & (rat_welch["sig_type"] == "neuron")  & (rat_welch["neuron_fr"].notnull()), 
    #                                     xr.where(rat_welch["neuron_fr"] < 2, "STR<2Hz", "STR>2Hz")
    #                                     ,rat_welch["neuron_type"])
    
    # for suffix in "_1", "_2":
    #     rat_coh["neuron_type"+suffix] = xr.where((rat_welch["structure"+suffix] == "STR") & (rat_welch["sig_type"] == "neuron")  & (rat_welch["neuron_fr"].notnull()), 
    #                                         xr.where(rat_welch["neuron_fr"] < 2, "STR<2Hz", "STR>2Hz")
    #                                         ,rat_welch["neuron_type"])
    # _save_xarray(rat_welch, base_result_path/"rat_welch.zarr")
    # _save_xarray(rat_coh, base_result_path/"rat_coh.zarr")
    monkey_welch = await process_monkey_data()
    human_welch = await process_human_data()
    #Probably some saving here too
    common_coords = ["f", "sig_type", "condition", "structure", "species", "neuron_type", "subject"]
    common_rat_welch = (
        rat_welch.assign_coords(species="Rat").where(~(rat_welch["has_swa"] | rat_welch["is_APO"]), drop=True)
        .drop_vars(k for k in rat_welch.coords if not k in common_coords)
        
    )

    common_monkey_welch = monkey_welch.assign_coords(species="Monkey", neuron_type="").drop_vars(k for k in monkey_welch.coords if not k in common_coords)
    common_monkey_welch["condition"] = xr.where(common_monkey_welch["condition"]=="healthy", "CTL", "Park")
    common_monkey_welch["structure"] = xr.where(common_monkey_welch["structure"]=="MSN", "STR", common_monkey_welch["structure"])

    common_human_welch = human_welch.assign_coords(species="human", neuron_type="").drop_vars(["session", "electrode", "id"])
    common_human_welch["structure"] = xr.where(common_human_welch["structure"]=="MSN", "STR", common_human_welch["structure"])
    common_human_welch["condition"] = xr.where(common_human_welch["structure"].str.contains("VMNR"), "CTL", common_human_welch["condition"])
    common_human_welch["structure"] = xr.where(common_human_welch["structure"].str.contains("STN"), "STN", common_human_welch["structure"])


    all_species_welch = xr.concat([common_monkey_welch, common_rat_welch, common_human_welch], dim="channel")

    paired_coords = ["sig_type", "structure", "neuron_type"]
    coord_pairs = [c+suffix for c in paired_coords for suffix in ["_1", "_2"]]
    unpaired_coords = [c for c in common_coords if not c in paired_coords]
    common_rat_coh = (
        rat_coh.assign_coords(species="Rat").where(~(rat_coh["has_swa"] | rat_coh["is_APO"]), drop=True)
        .drop_vars(k for k in rat_coh.coords if not k in unpaired_coords  and not k in coord_pairs)
    )

    all_species_coh = xr.concat([common_rat_coh], dim="channel_pair")

    for c in all_species_welch.coords:
        if len(all_species_welch[c].dims) ==0:
            all_species_welch[c] = all_species_welch[c].broadcast_like(all_species_welch["channel"])
        if all_species_welch[c].dtype==object:
            all_species_welch[c] = all_species_welch[c].astype(str)
    for c in all_species_coh.coords:
        # print(all_species_coh[c])
        # print(type(all_species_coh[c]))
        if len(all_species_coh[c].dims) ==0:
            all_species_coh[c] = all_species_coh[c].broadcast_like(all_species_coh["channel_pair"])
        if all_species_coh[c].dtype==object:
            all_species_coh[c] = all_species_coh[c].astype(str)
        

    _save_xarray(all_species_welch, base_result_path/"all_species_welch.zarr")
    _save_xarray(all_species_coh, base_result_path/"all_species_coh.zarr")

    errors = pd.read_csv(error_report.file, sep="\t")
    errors.to_excel("errors.xlsx")
    # all_species_welch["sig_type"] = all_species_welch["sig_type"].astype(object)
    # all_species_coh["sig_type_1"] = all_species_coh["sig_type_1"].astype(object)
    # all_species_welch.to_netcdf(base_result_path/"all_species_welch.h5", engine="netcdf4")
    
    # print(all_species_welch.to_dataset(name="welch").groupby(common_coords[1:]).apply(lambda d: xr.DataArray(d.sizes["channel"])))
if __name__ == "__main__":
    try:
        anyio.run(main, backend="trio")
    finally:
        pipeline_helper.close()