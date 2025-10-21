import xarray as xr, pandas as pd, numpy as np
import tqdm.auto as tqdm
from pathlib import Path
from typing import Literal, Dict, List, Union, Any
import re
from dafn.spike2 import smrxadc2electrophy, smrxchanneldata
from dafn.signal_processing import compute_lfp, compute_bua

def rat_raw_chan_classification(name: str):
    patterns = {
        r"^probe\d+$": "Probe",
        r"^.*contra.*eeg.*$": "EEGContra",
        r"^.*eeg.*contra.*$": "EEGContra",
        r"^.*eeg.*$": "EEGIpsi",
        r"^ecg": "ECG",
        r"^pinch": "Pinch",
        r"^stim": "Stim",
        r"^keyboard": "Keyboard",
        r"^respirtn": "Respirtn",
        r"^.*stn.*": "STN",
    }
    for pattern, label in patterns.items():
        if re.match(pattern, name):
            return label
        
def get_smrx_metadata(smrx_paths: List[Path]):
    ret_condition = {"Beta work":"Park", "Control work": "CTL"}

    spike2df = pd.DataFrame()
    spike2df["spike2_file"] = smrx_paths
    spike2df["condition"] = [ret_condition[p.parts[0]] for p in smrx_paths]
    spike2df["subject"] = [p.parts[1] for p in smrx_paths]
    spike2df["is_APO"] = spike2df["subject"].str.lower().str.contains("apo")
    spike2df["subject"] = spike2df["subject"].str.replace("(Apo)", "").str.replace(" ", "")

    def get_sessions_and_segment(stem):
        if stem.startswith("APE"):
            session = "a"
            segment_index = int(stem[-2:])
        else:
            l = re.findall(r'([a-z])(?:\.|\'| )?(\d\d?)', stem.lower())
            if len(l) > 0:
                session = l[-1][0]
                segment_index = int(l[-1][1])
            else:
                raise Exception(f"Unknown session segment for {stem}, candidates {l}")
        return session, segment_index
    session_and_segments = [get_sessions_and_segment(p.stem) for p in smrx_paths]

    spike2df["session"] = [t[0] for t in session_and_segments]
    spike2df["segment_index"] = [t[1] for t in session_and_segments]
    return spike2df

def get_matfile_metadata(mat_files: List[Path]):
    mat_df = pd.DataFrame()
    mat_df["mat_file"] = mat_files
    mat_df["mat_date"] = [f.parts[2] for f in mat_files]
    mat_df["condition"] = [f.parts[0] for f in mat_files]
    mat_df["structure"] = [f.parts[4] for f in mat_files]
    mat_df["subject"] = [f.parts[1] for f in mat_files]
    mat_df["session_info"] = [f.parts[3] for f in mat_files]
    mat_df["session_info"] = mat_df["session_info"].astype(str)
    mat_df["session"] = mat_df["session_info"].str.extract("(\D+)")
    mat_df["segment_index"] = mat_df["session_info"].str.extract("(\d+)").astype(int)
    mat_df["has_swa"] = mat_df["session_info"].str.contains("swa")
    mat_df["signal_type"] = [f.stem.lower().strip() for f in mat_files]

    mat_df["mat_date"] = pd.to_datetime(mat_df["mat_date"], format="%Y%m%d")
    mat_df = mat_df.drop(columns="session_info")
    return mat_df

def get_initial_signals(raw_file):
    all_chans = smrxchanneldata(raw_file)
    all_raw_chans = all_chans.loc[all_chans["physical_channel"]>= 0].copy()
    all_raw_chans["chan_group"] = all_raw_chans["chan_name"].str.lower().apply(rat_raw_chan_classification)
    probe_chan_nums = all_raw_chans["chan_num"].loc[all_raw_chans["chan_group"]=="Probe"].tolist()
    ipsieeg_chan_nums = all_raw_chans["chan_num"].loc[all_raw_chans["chan_group"]=="EEGIpsi"].tolist()
    if len(probe_chan_nums) == 0:
       return None
    if len(ipsieeg_chan_nums) != 1:
       raise Exception(f"Got {len(ipsieeg_chan_nums)} ipsi eeg")
    with smrxadc2electrophy(raw_file, probe_chan_nums) as raw:
      lfp = compute_lfp(raw)
      bua = compute_bua(raw)
      with smrxadc2electrophy(raw_file, ipsieeg_chan_nums) as raw_eeg:
        eeg = compute_lfp(raw_eeg)
        return lfp, bua, eeg