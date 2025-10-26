# %%
import plotly.express as px
from pathlib import Path
import xarray as xr, pandas as pd, numpy as np
from pipeline_helper import _load_xarray
from dask.diagnostics import ProgressBar
from dafn.plot_utilities import plotly_config, faceted_imshow_xarray, line_error_bands


# %%
structure_order = ["GPe", "STN", "STR", "eeg"]
condition_order = ["Park", "CTL"]

# %%
base_folder = Path("/media/julienb/T7 Shield/Revue-FRM/AnalysisData/")
all_welch =xr.open_zarr(base_folder/"all_welch.zarr", chunks=None).sel(f=slice(3, 55))["data"]
all_welch["structure"] = xr.where(all_welch["sig_type"] == "eeg", "eeg", all_welch["structure"])
welch_groups = ["condition", "structure", "sig_type", "is_APO", "has_swa"]
for k in welch_groups:
    all_welch[k] = all_welch[k].broadcast_like(all_welch["channel"])
welch_group = all_welch.to_dataset(name="welch").groupby(welch_groups)


all_welch_grouped = welch_group.mean()["welch"]
all_welch_grouped["n_sigs"] = welch_group.map(lambda x: xr.DataArray(x.sizes["channel"])).fillna(0).astype(int)
all_welch_grouped["n_subjects"] = welch_group.apply(lambda x: xr.DataArray(len(np.unique(x["subject"])))).fillna(0).astype(int)
all_welch_grouped["n_segments"] = welch_group.apply(
    lambda x: xr.DataArray(len(np.unique(np.stack([x["session"], x["session_grp"], x["subject"]], axis=1), axis=0)))).fillna(0).astype(int)
all_welch_grouped["n_session_grp"] = welch_group.apply(
    lambda x: xr.DataArray(len(np.unique(np.stack([x["session_grp"], x["subject"]], axis=1), axis=0)))).fillna(0).astype(int)
all_welch_grouped["sem"] = welch_group.std()["welch"] / np.sqrt(all_welch_grouped["n_sigs"])
all_welch_grouped.to_dataset()


# %%
plot_welch = all_welch_grouped.sel(is_APO=False)
plot_welch = plot_welch.where(plot_welch.notnull(), drop=True)
fig = line_error_bands(plot_welch.to_dataframe(name="welch").reset_index(), 
              x="f", y="welch", color="condition", facet_row="structure", facet_col="sig_type", error_y="sem",
              hover_data=["n_sigs","n_segments", "n_session_grp", "n_subjects", "sem"], height=800, title="Pwelch for Rats",
              category_orders=dict(condition=condition_order, structure=structure_order))
fig.show(config=plotly_config)
fig.write_html("pwelch.html", config=plotly_config)

# %%
all_coh = xr.open_zarr(base_folder/"all_coh.zarr", chunks=None).sel(f=slice(3, 55))["data"]
all_coh["structure_1"] = xr.where(all_coh["sig_type_1"] == "eeg", "eeg", all_coh["structure_1"])
all_coh["structure_2"] = xr.where(all_coh["sig_type_2"] == "eeg", "eeg", all_coh["structure_2"])

coh_groups = ["condition", "structure_1", "structure_2", "sig_type_1", "sig_type_2", "is_APO", "has_swa"]
for k in coh_groups:
    all_coh[k] = all_coh[k].broadcast_like(all_coh["channel_pair"])
coh_group = all_coh.to_dataset(name="coh").groupby(coh_groups)

all_coh_grouped = coh_group.mean()["coh"]
all_coh_grouped["n_sigs"] = coh_group.map(lambda x: xr.DataArray(x.sizes["channel_pair"])).fillna(0).astype(int)
all_coh_grouped["n_subjects"] = coh_group.apply(lambda x: xr.DataArray(len(np.unique(x["subject"])))).fillna(0).astype(int)
all_coh_grouped["n_segments"] = coh_group.apply(
    lambda x: xr.DataArray(len(np.unique(np.stack([x["session"], x["session_grp"], x["subject"]], axis=1), axis=0)))).fillna(0).astype(int)
all_coh_grouped["n_session_grp"] = coh_group.apply(
    lambda x: xr.DataArray(len(np.unique(np.stack([x["session_grp"], x["subject"]], axis=1), axis=0)))).fillna(0).astype(int)
all_coh_grouped["sem"] = np.abs(coh_group.std()["coh"] / np.sqrt(all_coh_grouped["n_sigs"]))
all_coh_grouped.to_dataset()


# %%
plot_coh_power = np.abs(all_coh_grouped).sel(is_APO=False, has_swa=0, sig_type_1=["bua", "eeg"], sig_type_2=["bua", "eeg"])
plot_coh_power = plot_coh_power.where(plot_coh_power.notnull(), drop=True)
fig = line_error_bands(plot_coh_power.to_dataframe(name="coh").reset_index().dropna(subset="coh"), 
              x="f", y="coh", color="condition", facet_row="structure_2", facet_col="structure_1", error_y="sem",
              category_orders=dict(condition=condition_order, structure_1=structure_order, structure_2=structure_order[:-1]),
              hover_data=["n_sigs","n_segments", "n_session_grp", "n_subjects", "sem"], title="BUA Coherence power for Rats", height=800)
fig.show(config=plotly_config)
fig.write_html("coh_power.html", config=plotly_config)

# %%
phase_coh = all_coh.sel(f=20)
coh = xr.Dataset()
coh["phase"] = xr.apply_ufunc(lambda a: np.angle(a, deg=True) + 180,phase_coh)
coh["power"] = np.abs(phase_coh)
tmp = []
for _, g in coh["power"].groupby(coh_groups):
    r = (g-g.mean())/g.std()
    r = r-r.min()
    tmp.append(r)
coh["power"] = xr.concat(tmp, dim="channel_pair")
heatmap_groups = coh.groupby(dict(phase=xr.groupers.BinGrouper(np.linspace(0, 360, 13)), power=xr.groupers.BinGrouper(bins=8)) 
                            | {k: xr.groupers.UniqueGrouper() for k in coh_groups})
coh_phase_groups = heatmap_groups.apply(lambda x: xr.DataArray(x.sizes["channel_pair"])).fillna(0).astype(int)
coh_phase_groups["density"] = coh_phase_groups / coh_phase_groups.sum(["phase_bins", "power_bins"])
coh_phase_groups["phase_bins_mid"] = xr.apply_ufunc(lambda x: x.mid,coh_phase_groups["phase_bins"], vectorize=True)
coh_phase_groups["power_bins_right"] = xr.apply_ufunc(lambda x: x.right,coh_phase_groups["power_bins"], vectorize=True)
coh_phase_groups = coh_phase_groups.assign_coords({k: all_coh_grouped[k] for k in ["n_sigs", "n_subjects", "n_segments", "n_session_grp"]})
coh_phase_groups.to_dataset(name="count")




# %%
plot_coh_phase = coh_phase_groups.sel(is_APO=False, has_swa=0, sig_type_1=["bua", "eeg"], sig_type_2=["bua", "eeg"], condition="Park")
plot_coh_phase = plot_coh_phase.stack(dim1=("structure_1", "sig_type_1"), dim2=("structure_2", "sig_type_2"), create_index=False)
plot_coh_phase = plot_coh_phase.where(plot_coh_phase["n_sigs"]>20, drop=True)
plot_coh_phase = plot_coh_phase.where((plot_coh_phase > 0).any(["phase_bins"]), drop=True)
plot_coh_phase = plot_coh_phase.swap_dims(dim1="structure_1", dim2="structure_2", power_bins="power_bins_right", phase_bins="phase_bins_mid")
fig = faceted_imshow_xarray(plot_coh_phase["density"], r_dim="power_bins_right", theta_dim="phase_bins_mid", facet_row="structure_2",facet_col="structure_1", subplot_width=290, subplot_height=250)
fig.update_layout(title="BUA Coherence phase at 20Hz for Rats")
fig.show(config=plotly_config)
fig.write_html("coh_phase.html", config=plotly_config)


# %%

# coh = coh.assign_coords(nsigs=n_sigs)
# coh = coh.where(coh["nsigs"]>20, drop=True)
# coh["power"] = coh["power"]/coh["power"].groupby(groups).mean()

coh["count"] = xr.ones_like(coh["power"])
phase_density = coh.groupby(dict(phase=xr.groupers.BinGrouper(np.linspace(0, 360, 13)), power=xr.groupers.BinGrouper(bins=8)) 
                            | {k: xr.groupers.UniqueGrouper() for k in groups}).sum()
phase_density["count"] =  phase_density["count"]/phase_density["count"].sum(["phase_bins", "power_bins"])
phase_density["phase_bins"] = [x.mid for x in phase_density["phase_bins"].data]
phase_density["power_bins"] = [x.right for x in phase_density["power_bins"].data]
phase_density = phase_density.assign_coords(nsigs=n_sigs)
phase_density


# %%
from dafn.plot_utilities import faceted_imshow_xarray
ar = 1*((phase_density["count"]).sel(sig_type_1="bua", sig_type_2="bua", is_APO=False, condition="Park").fillna(0))
ar = ar.where(ar["nsigs"]>20, drop=True)
ar = ar.where((ar > 0).any(["phase_bins"]), drop=True)
# ar = ar.fillna(0)
fig = faceted_imshow_xarray(ar, r_dim="power_bins", theta_dim="phase_bins", facet_row="structure_2",facet_col="structure_1", subplot_width=250, subplot_height=250)
fig.show(config=plotly_config)


