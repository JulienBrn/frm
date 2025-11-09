import numpy as np, pandas as pd, xarray as xr
from unionfind import unionfind
import dask.array as da

def get_electrode_groups(df: pd.DataFrame):
    n = len(df)
    if n == 0:
        return pd.Series([], dtype=int, index=df.index)

    uf = unionfind(n)

    for i in range(n):
        for j in range(i + 1, n):
            if df["start"].iat[i]-1.5 >= df["start"].iat[j] and df["start"].iat[i]+1.5<= df["end"].iat[j]:
                uf.unite(i, j)
            elif df["start"].iat[j] -1.5 >= df["start"].iat[i] and df["start"].iat[j]+1.5<= df["end"].iat[i]:
                uf.unite(i, j)
    groups = [-1]*n
    for l in uf.groups():
      start = min([df["start"].iloc[k] for k in l])
      for k in l:
        groups[k]=start
    return pd.Series(groups, index=df.index)

def make_raw_chan(df: pd.DataFrame, monkey_basefolder, fs=25000):
    def get_array(row):
        from scipy.io import loadmat
        try:
          data: np.ndarray = loadmat(monkey_basefolder / row["file_path"])["RAW"][0]
        except Exception:
            print(f"Problem loading data {row['file_path']}")
            raise
        if data.ndim != 1:
            raise Exception("Data with more than a single dim...")
        start_ind = int(row["start"]*fs)
        return (start_ind, start_ind+data.size, data)
    
    all_arrays = [get_array(row) for _, row in df.iterrows()]
    
    def get_shifts(all_arrays):
      n_arrs = len(all_arrays)
      shifts = np.full((n_arrs, n_arrs), None, dtype=object)
      from dafn.utilities import get_subsequence_positions
      for i in range(n_arrs):
          for j in range(i+1, n_arrs):
              start_i, end_i, data_i = all_arrays[i]
              start_j, end_j, data_j = all_arrays[j]
              
              common_slice_start = max(start_i, start_j)
              common_slice_end   = min(end_i, end_j)

              if common_slice_start >= common_slice_end - 3*fs:
                  continue  # not enough overlap
              
              to_search = data_i[common_slice_start-start_i+fs: common_slice_end-start_i-fs]
              search_into = data_j[common_slice_start-start_j: common_slice_end-start_j]
              indices = get_subsequence_positions(to_search, search_into)
              if len(indices) == 1: #Exactly one position found
                  shifts[i, j] = indices[0] - (common_slice_start-start_i+fs) + (common_slice_start-start_j)
      return shifts
    
    shifts = get_shifts(all_arrays)

    def resolve_real_starts(starts, shifts, ref=0):
      from collections import deque
      n = shifts.shape[0]
      offsets = np.full(n, None, dtype=object)
      offsets[ref] = starts[ref]

      visited = set([(ref, ref)])
      queue = deque([ref])

      while queue:
          i = queue.popleft()
          for j in range(n):
              if (i, j) in visited:
                  continue
              if not shifts[i, j] is None:
                  expected_offset = offsets[i] - shifts[i, j]
                  if not offsets[j] is None and expected_offset != offsets[j]:
                      raise Exception("Different computed offsets")
                  offsets[j] = expected_offset
                  visited.add((i, j))
                  queue.append(j)
      if (offsets==None).any():
          raise Exception("problem")
      return offsets.astype(int)
    
    new_starts = resolve_real_starts([s for s, e, d in all_arrays], shifts)
    final_start = min(new_starts)
    final_end = max([s+d.size for s, (_, _, d) in zip(new_starts, all_arrays)])
    res = np.full(final_end-final_start, 0, dtype=all_arrays[0][2].dtype)
    written_to = np.full(final_end-final_start, 0, dtype=bool)
    for s, (_, _, d) in zip(new_starts, all_arrays):
        rs = s-final_start
        if (np.abs(res[rs:rs+d.size][written_to[rs:rs+d.size]] - d[written_to[rs:rs+d.size]]) > 10**-6).any():
            raise Exception("Inconsistent data")
        res[rs:rs+d.size] = d
        written_to[rs:rs+d.size] = True
    final = xr.DataArray(da.array(res), dims="t")
    final["t"] = (np.arange(res.size) + final_start)/fs
    final["t"].attrs["fs"] = fs
    return final