import xarray as xr, pandas as pd, numpy as np
import tqdm.auto as tqdm
from pathlib import Path
from typing import Literal, Dict, List, Union, Any
import re, h5py
from dafn.spike2 import smrxadc2electrophy, smrxchanneldata, sonfile
from dafn.signal_processing import compute_lfp, compute_bua
import logging
logger = logging.getLogger(__name__)

def print_h5_tree(name, obj):
    if isinstance(obj, h5py.Dataset):
        print(f"Dataset: {name}, shape: {obj.shape}, dtype: {obj.dtype}")
    elif isinstance(obj, h5py.Group):
        print(f"Group: {name}")

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

# def get_initial_signals(raw_file):
#     all_chans = smrxchanneldata(raw_file)
#     all_raw_chans = all_chans.loc[all_chans["physical_channel"]>= 0].copy()
#     all_raw_chans["chan_group"] = all_raw_chans["chan_name"].str.lower().apply(rat_raw_chan_classification)
#     probe_chan_nums = all_raw_chans["chan_num"].loc[all_raw_chans["chan_group"]=="Probe"].tolist()
#     ipsieeg_chan_nums = all_raw_chans["chan_num"].loc[all_raw_chans["chan_group"]=="EEGIpsi"].tolist()
#     if len(probe_chan_nums) == 0:
#        return None
#     if len(ipsieeg_chan_nums) != 1:
#        raise Exception(f"Got {len(ipsieeg_chan_nums)} ipsi eeg")
#     with smrxadc2electrophy(raw_file, probe_chan_nums) as raw:
#       lfp = compute_lfp(raw)
#       bua = compute_bua(raw)
#       with smrxadc2electrophy(raw_file, ipsieeg_chan_nums) as raw_eeg:
#         eeg = compute_lfp(raw_eeg)
#         return lfp, bua, eeg
      
def get_mat_df(xr_mats: xr.DataArray, base_folder):
    df = xr_mats.sel(signal_type="raw", drop=True).to_dataframe(name="mat_path").reset_index()
    df = df.loc[df["mat_path"]!=""]
    rows = []
    for _, row in df.iterrows():
        file_path = row['mat_path']
        with h5py.File(base_folder/file_path, 'r') as mat:
            for key in mat.keys():
                match = re.search(r'_ch(?P<num>\d+)$', key.lower())
                channel_num = int(match.group("num")) - 1 if match else pd.NA
                rows.append({
                    'mat_path': file_path,
                    'mat_key': key,
                    'chan_num': channel_num
                })
    res =  pd.DataFrame(rows, columns=["mat_path", "mat_key", "chan_num"]).merge(df, how="left", on="mat_path")
    res["chan_num"] = res["chan_num"].astype('Int64')
    return res

def mat_spike2_raw_join(mat_df: pd.DataFrame, raw_df: pd.DataFrame) -> pd.DataFrame:
    endswith_vec = np.frompyfunc(lambda a, b: a.endswith(b), 2, 1)

    chan_num_mat = mat_df["chan_num"].to_numpy()
    chan_key_mat = mat_df["mat_key"].astype(str).to_numpy()

    chan_num_raw = raw_df["chan_num"].to_numpy()
    chan_name_raw = raw_df["chan_name"].astype(str).to_numpy()

    equals_1 = chan_num_mat[:, None] == chan_num_raw[None, :]
    equals_2 = chan_key_mat[:, None] == chan_name_raw[None, :]
    equals_3 = endswith_vec(chan_key_mat[:, None], chan_name_raw[None, :]).astype(bool)
    equals = equals_1 | equals_2 | equals_3

    if (equals.sum(axis=0) > 1).any():
        raw_idx = np.flatnonzero((equals.sum(axis=0) > 1))
        # print(raw_df.iloc[raw_idx])
        # print(mat_df)
        raise Exception("A raw matches several mats")
    if (equals.sum(axis=1) > 1).any():
        # print(raw_df)
        # print(mat_df)
        raise Exception("A mat matches several raw")
    if (equals.sum(axis=1) == 0).any():
        mat_idx = np.flatnonzero((equals.sum(axis=1) == 0))
        # print(mat_df.iloc[mat_idx])
        # print(raw_df)
        # logger.warning("Missing matches")

    mat_idx, raw_idx = np.where(equals & (equals.sum(axis=1)==1)[:, None]  & (equals.sum(axis=0)==1)[None, :])
    matched = mat_df.iloc[mat_idx].drop(columns="chan_num").reset_index(drop=True).join(
        raw_df.iloc[raw_idx].reset_index(drop=True))

    return matched

def get_subsequence_positions(sub, a, tol=10**(-6)):
    candidates = np.ones(a.size, dtype=bool)
    i= 0
    while i<sub.size:
        candidates = candidates & (np.abs(np.roll(a, -i) - sub[i]) < tol)
        sum = candidates.sum()
        if sum < 50:
            break
        i+=1
            
    candidates = np.flatnonzero(candidates)
    res = []
    for i in candidates:
        if (a.size - i) < sub.size:
            break
        if (np.abs(a[i:i+sub.size] - sub) < tol).all():
            res.append(i)
    return res

def extract_timing(joined_df: pd.DataFrame, spike2_file, mat_basefolder):
    joined_df = joined_df.dropna(subset="mat_key")
    chan_start_times = []
    chan_end_times = []
    with sonfile(spike2_file) as rec:
        time_base = rec.GetTimeBase()
        for _, row in joined_df.iterrows():
            key = row["mat_key"]
            file = row["mat_path"]
            chan_num = row["chan_num"]
            chan_name = row["chan_name"]
            with h5py.File(mat_basefolder / file, 'r') as f:
                data_size = f[key]["values"].size
                fs = 1.0/float(f[key]["interval"][0,0])
                duration = data_size/fs
                subarray = np.array(f[key]["values"][0, :1000])
            j = 0
            chunk_size = 10**7
            divide = rec.ChannelDivide(chan_num)
            positions = []
            while True:
                chunk = np.array(rec.ReadFloats(chan_num, chunk_size, int(divide*j*(chunk_size-subarray.size))))
                ps = get_subsequence_positions(subarray, chunk)
                positions+=[int(p+j*(chunk_size-subarray.size)) for p in ps]
                if len(chunk) < chunk_size:
                    break
                j+=1
            times = [p*(divide*time_base) for p in positions]
            if len(times) != 1:
                raise Exception(f"{len(times)} candidates for period of interest. Channel name is {chan_name}")
            chan_start_times.append(times[0])
            chan_end_times.append(times[0]+duration)
    for chan_times in [chan_start_times, chan_end_times]:
        chan_times = np.array(chan_times)
        if (np.abs(chan_times - chan_times.mean()) > 10**-4).any():
            print(chan_times)
            raise Exception("Not all same times")
    return np.array(chan_start_times).mean(), np.array(chan_end_times).mean()