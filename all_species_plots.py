# %%
import plotly.express as px
from pathlib import Path
import xarray as xr, pandas as pd, numpy as np
from pipeline_helper import _load_xarray
from dask.diagnostics import ProgressBar
from dafn.plot_utilities import plotly_config, faceted_imshow_xarray, line_error_bands
import itables
itables.init_notebook_mode(all_interactive=True)

# %%
sig_group_order = ["eeg", "STR", "STN", "GPe", "Arky", "Proto"]
condition_order = ["Park", "CTL"]

# %%
show = False
base_folder = Path("/media/julienb/T7 Shield/Revue-FRM/AnalysisData8/")
result_folder = Path("./all_species")
result_folder.mkdir(exist_ok=True, parents=True)
all_welch =_load_xarray(base_folder/"all_species_welch.zarr").sel(f=slice(3, 55))
# all_welch =xr.open_dataarray(base_folder/"all_species_welch.h5").sel(f=slice(3, 55))
all_welch["sig_group"] = xr.where(all_welch["sig_type"] == "eeg", "eeg", all_welch["structure"])
all_welch["sig_group"] = xr.where(all_welch["neuron_type"]!="", all_welch["neuron_type"], all_welch["sig_group"])
# display(all_welch)
welch_groups = ["condition", "sig_group", "sig_type", "species"]
welch_group = all_welch.to_dataset(name="welch").groupby(welch_groups)


all_welch_grouped = welch_group.mean()["welch"]
all_welch_grouped["n_sigs"] = welch_group.map(lambda x: xr.DataArray(x.sizes["channel"])).fillna(0).astype(int)
all_welch_grouped["n_subjects"] = welch_group.apply(lambda x: xr.DataArray(len(np.unique(x["subject"])))).fillna(0).astype(int)
all_welch_grouped["sem"] = welch_group.std()["welch"] / np.sqrt(all_welch_grouped["n_sigs"])
display(all_welch_grouped.to_dataset())
all_welch_grouped["n_sigs"].to_dataframe()

# %%
for st in ["eeg", "lfp", "bua", "neuron"]:
  plot_welch = all_welch_grouped.sel(sig_type=[st])
  plot_welch = plot_welch.where(plot_welch.notnull(), drop=True)

  fig = line_error_bands(plot_welch.to_dataframe(name="welch").reset_index(), 
                x="f", y="welch", color="condition", facet_row="sig_group", facet_col="species", error_y="sem",
                hover_data=["n_sigs", "n_subjects", "sem"], height=800, title=f"{st.upper()} Pwelch",
                category_orders=dict(condition=condition_order, sig_group=[sg for sg in sig_group_order if sg in plot_welch["sig_group"]]))
  if show:
    fig.show(config=plotly_config)
  fig.write_html(result_folder/f"pwelch_compare_cond_{st}.html", config=plotly_config)

  fig = line_error_bands(plot_welch.sel(condition="Park").to_dataframe(name="welch").reset_index(), 
                x="f", y="welch", color="species", facet_row="condition", facet_col="sig_group", error_y="sem",
                hover_data=["n_sigs", "n_subjects", "sem"], height=400, title=f"{st.upper()} Pwelch Park",
                category_orders=dict(sig_group=[sg for sg in sig_group_order if sg in plot_welch["sig_group"]]))
  if show:
    fig.show(config=plotly_config)
  fig.write_html(result_folder/f"pwelch_compare_species_{st}.html", config=plotly_config)

  fig = line_error_bands(plot_welch.sel(condition="Park").to_dataframe(name="welch").reset_index(), 
                x="f", y="welch", color="sig_group", facet_row="condition", facet_col="species", error_y="sem",
                hover_data=["n_sigs", "n_subjects", "sem"], height=400, title=f"{st.upper()} Pwelch Park",
                category_orders=dict(csig_group=[sg for sg in sig_group_order if sg in plot_welch["sig_group"]]))
  if show:
    fig.show(config=plotly_config)
  fig.write_html(result_folder/f"pwelch_compare_structure_{st}.html", config=plotly_config)

# %%
all_coh =  _load_xarray(base_folder/"all_species_coh.zarr").sel(f=slice(3, 55))
for suffix in ["_1", "_2"]:
  all_coh["sig_group"+suffix] = xr.where(all_coh["sig_type"+suffix] == "eeg", "eeg", all_coh["structure"+suffix])
  all_coh["sig_group"+suffix] = xr.where(all_coh["neuron_type"+suffix] != "", all_coh["neuron_type"+suffix], all_coh["sig_group"+suffix])

coh_groups = ["species", "condition", "sig_group_1", "sig_group_2", "sig_type_1", "sig_type_2"]
for k in coh_groups:
    all_coh[k] = all_coh[k].broadcast_like(all_coh["channel_pair"])
coh_group = np.abs(all_coh).to_dataset(name="coh").groupby(coh_groups)

all_coh_grouped = coh_group.mean()["coh"]
all_coh_grouped["n_sigs"] = coh_group.map(lambda x: xr.DataArray(x.sizes["channel_pair"])).fillna(0).astype(int)
all_coh_grouped["n_subjects"] = coh_group.apply(lambda x: xr.DataArray(len(np.unique(x["subject"])))).fillna(0).astype(int)
# all_coh_grouped["n_segments"] = coh_group.apply(
#     lambda x: xr.DataArray(len(np.unique(np.stack([x["session"], x["session_grp"], x["subject"]], axis=1), axis=0)))).fillna(0).astype(int)
# all_coh_grouped["n_session_grp"] = coh_group.apply(
#     lambda x: xr.DataArray(len(np.unique(np.stack([x["session_grp"], x["subject"]], axis=1), axis=0)))).fillna(0).astype(int)
all_coh_grouped["sem"] = np.abs(coh_group.std()["coh"] / np.sqrt(all_coh_grouped["n_sigs"]))
all_coh_grouped.to_dataset()


# %%
for species in all_coh_grouped["species"].to_numpy():
  for st in ["lfp", "bua", "neuron"]:
    plot_coh_power = np.abs(all_coh_grouped).sel(sig_type_1=[st, "eeg"], sig_type_2=[st, "eeg"])
    plot_coh_power = plot_coh_power.where(plot_coh_power.notnull(), drop=True)
    fig = line_error_bands(plot_coh_power.sel(species=species).to_dataframe(name="coh").reset_index().dropna(subset="coh"), 
                  x="f", y="coh", color="condition", facet_row="sig_group_2", facet_col="sig_group_1", error_y="sem",
                  category_orders=dict(condition=[c for c in condition_order if c in plot_coh_power["condition"]], 
                                      sig_group_1=[sg for sg in sig_group_order if sg in plot_coh_power["sig_group_1"]],
                                      sig_group_2=[sg for sg in sig_group_order if sg in plot_coh_power["sig_group_2"]]),
                  hover_data=["n_sigs", "n_subjects", "sem"], title=f"{st.upper()} Coherence power for {species}", height=800)
    if show:
      fig.show(config=plotly_config)
    fig.write_html(result_folder/f"coh_power_{species}_{st}.html", config=plotly_config)

    subplot = plot_coh_power.sel(species=species, sig_group_2="eeg", condition="Park", sig_type_1=[st])
    fig = line_error_bands(subplot.to_dataframe(name="coh").reset_index().dropna(subset="coh"), 
                  x="f", y="coh", color="sig_group_1", facet_row="sig_group_2", error_y="sem",
                  category_orders=dict(condition=[c for c in condition_order if c in subplot["condition"]], 
                                      sig_group_1=[sg for sg in sig_group_order if sg in subplot["sig_group_1"]],
                                      sig_group_2=[sg for sg in sig_group_order if sg in subplot["sig_group_2"]]),
                  hover_data=["n_sigs", "n_subjects", "sem"], title=f"{st.upper()} Coherence power for {species} Park rel EEG", height=400, width=600)
    if show:
      fig.show(config=plotly_config)
    fig.write_html(result_folder/f"eeg_park_coh_power_{species}_{st}.html", config=plotly_config)

# %%
for species in all_coh_grouped["species"].to_numpy():
  for st in ["lfp", "bua", "neuron"]:
    phase_coh = xr.concat([all_coh.sel(f=20).assign_coords(fmethod="20Hz"), 
                          all_coh.sel(f=np.abs(all_coh).sel(f=slice(8, 35)).idxmax("f")).assign_coords(fmethod="maxf")], dim="fmethod")
    angles = xr.DataArray(np.linspace(-np.pi, np.pi, 100, endpoint=False), dims="angle")
    angles["angle"] = angles * 180/np.pi
    angle_img = np.exp(angles*1j)
    def compute_value(a, angle_img):
        z = a*angle_img
        res= np.nansum(np.abs(z)*np.exp(-10*np.angle(z)**2))
        return res
    res = phase_coh.groupby(coh_groups).apply(lambda g: xr.apply_ufunc(compute_value, g, angle_img, input_core_dims=[["channel_pair"], []], vectorize=True))
    res = res/res.sum("angle")
    res = res.assign_coords({k: all_coh_grouped[k] for k in ["n_sigs", "n_subjects"]})
    res = res.sel(sig_type_1=[st, "eeg"], sig_type_2=[st, "eeg"], condition="Park")

    res= res.where(res.notnull(), drop=True)
    # display(res.notnull().groupby("sig_group_2").apply(lambda d: d.sum()))
    # display(res)
    f1 = px.line(res.sel(species=species).to_dataframe(name="power").reset_index().dropna(), 
            y="power", x="angle", facet_row="sig_group_2", facet_col="sig_group_1", color="fmethod",
            category_orders=dict(sig_group_1=[sg for sg in sig_group_order if sg in res["sig_group_1"]],
                                        sig_group_2=[sg for sg in sig_group_order if sg in res["sig_group_2"]]),
            hover_data=["n_sigs", "n_subjects"], height=800,
            title=f"{st.upper()} Coherence phase distribution for {species}")
    f1.write_html(result_folder/f"coh_phase_distribution_{st}_{species}_details.html", config=plotly_config)
    display(f1)
    f2 = px.line(res.sel(sig_group_1="eeg", species=species).to_dataframe(name="power").reset_index().dropna(), 
            y="power", x="angle", color="sig_group_2", facet_col = "fmethod",
            category_orders=dict(sig_group_1=[sg for sg in sig_group_order if sg in res["sig_group_1"]],
                                        sig_group_2=[sg for sg in sig_group_order if sg in res["sig_group_2"]]),
            hover_data=["n_sigs","n_subjects"],
            title=f"{st.upper()} Coherence phase distribution at 20Hz for {species} relative to EEG")
    f2.write_html(result_folder/f"coh_phase_distribution_relEEG__{species}_{st}.html", config=plotly_config)
    display(f2)


