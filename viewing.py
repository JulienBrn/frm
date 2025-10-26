# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: ss
#     language: python
#     name: python3
# ---

import plotly.express as px
from pathlib import Path
import xarray as xr, pandas as pd, numpy as np
from pipeline_helper import _load_xarray
from dask.diagnostics import ProgressBar


base_folder = Path("/media/julienb/T7 Shield/Revue-FRM/AnalysisData/")
all_welch =xr.open_zarr(base_folder/"all_welch.zarr", chunks=None).sel(f=slice(3, 55))["data"]
all_welch = all_welch.groupby(["condition", "structure", "sig_type", "is_APO"]).mean().compute()
all_welch.to_dataset()


plot_welch = all_welch.sel(sig_type = "bua", is_APO=False)
fig = px.line(plot_welch.to_dataframe(name="welch").reset_index(), x="f", y="welch", line_dash="condition", color="structure", facet_row="sig_type")
fig

# +
with ProgressBar():
  full_all_coh = xr.open_zarr(base_folder/"all_coh.zarr", chunks=None).sel(f=slice(3, 55))["data"]
  full_all_coh["structure_1"] = xr.where(full_all_coh["sig_type_1"] == "eeg", "eeg", full_all_coh["structure_1"])
  full_all_coh["structure_2"] = xr.where(full_all_coh["sig_type_2"] == "eeg", "eeg", full_all_coh["structure_2"])
  full_all_coh["sig_type_1"] = xr.where(full_all_coh["sig_type_1"] == "eeg", "bua", full_all_coh["sig_type_1"])
  full_all_coh["sig_type_2"] = xr.where(full_all_coh["sig_type_2"] == "eeg", "bua", full_all_coh["sig_type_2"])
  full_all_coh = full_all_coh.where((full_all_coh["sig_type_1"] != full_all_coh["sig_type_2"]) | (full_all_coh["structure_1"] != full_all_coh["structure_2"]), drop=True)

groups = ["condition", "structure_1", "structure_2", "sig_type_1", "sig_type_2", "is_APO"]

n_sigs = full_all_coh.groupby(groups)["condition"].count()

# +


power_coh = np.abs(full_all_coh).groupby(groups).mean().compute()
power_coh = power_coh.sel(sig_type_1 = "bua", sig_type_2="bua", is_APO=False)
power = px.line(power_coh.to_dataframe(name="power").reset_index(), x="f", y="power", color="condition", facet_row="structure_1", facet_col="structure_2", height=800)
display(power)


# +
phase_coh = full_all_coh.sel(f=20)
coh = xr.Dataset()
coh["phase"] = xr.apply_ufunc(lambda a: np.angle(a, deg=True) + 180,phase_coh)
coh["power"] = np.abs(phase_coh)
tmp = []
for _, g in coh["power"].groupby(groups):
    r = (g-g.mean())/g.std()
    r = r-r.min()
    tmp.append(r)
coh["power"] = xr.concat(tmp, dim="channel_pair")
# coh["power"] = coh["power"]/coh["power"].groupby(groups).mean()

coh["count"] = xr.ones_like(coh["power"])
phase_density = coh.groupby(dict(phase=xr.groupers.BinGrouper(np.linspace(0, 360, 40)), power=xr.groupers.BinGrouper(bins=10)) 
                            | {k: xr.groupers.UniqueGrouper() for k in groups}).sum()
phase_density["count"] =  phase_density["count"]/phase_density["count"].sum(["phase_bins", "power_bins"])
phase_density["phase_bins"] = [x.mid for x in phase_density["phase_bins"].data]
phase_density["power_bins"] = [x.right for x in phase_density["power_bins"].data]
phase_density

# -

from dafn.plot_utilities import faceted_imshow_xarray
ar = 1*((phase_density["count"]).sel(sig_type_1="bua", sig_type_2="bua", is_APO=False, condition="Park").fillna(0))
fig = faceted_imshow_xarray(ar, r_dim="power_bins", theta_dim="phase_bins", facet_row="structure_2",facet_col="structure_1", subplot_width=250, subplot_height=250)
display(fig)
