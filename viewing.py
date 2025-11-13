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
sig_group_order = ["eeg", "GPe", "STN", "STR", "Arky", "Proto"]
condition_order = ["Park", "CTL"]

# %%
base_folder = Path("/media/julienb/T7 Shield/Revue-FRM/AnalysisData6/")
all_welch =_load_xarray(base_folder/"rat_welch.zarr").sel(f=slice(3, 55))
display(np.unique(all_welch["is_APO"].to_numpy()))
all_welch["sig_group"] = xr.where(all_welch["sig_type"] == "eeg", "eeg", all_welch["structure"])
all_welch["sig_group"] = xr.where(all_welch["neuron_type"]!="", all_welch["neuron_type"], all_welch["sig_group"])
display(all_welch)
welch_groups = ["condition", "sig_group", "sig_type", "is_APO", "has_swa"]
print(all_welch.isnull().sum())
# for k in welch_groups:
#     all_welch[k] = all_welch[k].broadcast_like(all_welch["channel"])
welch_group = all_welch.to_dataset(name="welch").groupby(welch_groups)


all_welch_grouped = welch_group.mean()["welch"]
all_welch_grouped["n_sigs"] = welch_group.map(lambda x: xr.DataArray(x.sizes["channel"])).fillna(0).astype(int)
all_welch_grouped["n_subjects"] = welch_group.apply(lambda x: xr.DataArray(len(np.unique(x["subject"])))).fillna(0).astype(int)
all_welch_grouped["n_segments"] = welch_group.apply(
    lambda x: xr.DataArray(len(np.unique(np.stack([x["session"], x["session_grp"], x["subject"]], axis=1), axis=0)))).fillna(0).astype(int)
all_welch_grouped["n_session_grp"] = welch_group.apply(
    lambda x: xr.DataArray(len(np.unique(np.stack([x["session_grp"], x["subject"]], axis=1), axis=0)))).fillna(0).astype(int)
all_welch_grouped["sem"] = welch_group.std()["welch"] / np.sqrt(all_welch_grouped["n_sigs"])
display(all_welch_grouped.to_dataset())
all_welch_grouped["n_sigs"].to_dataframe()


# %%
plot_welch = all_welch_grouped.sel(is_APO=False, has_swa=False, sig_type=["eeg", "lfp", "bua", "neuron"])
plot_welch = plot_welch.where(plot_welch.notnull(), drop=True)
fig = line_error_bands(plot_welch.to_dataframe(name="welch").reset_index(), 
              x="f", y="welch", color="condition", facet_row="sig_group", facet_col="sig_type", error_y="sem",
              hover_data=["n_sigs","n_segments", "n_session_grp", "n_subjects", "sem"], height=800, title="Pwelch for Rats",
              category_orders=dict(condition=condition_order, sig_group=[sg for sg in sig_group_order if sg in plot_welch["sig_group"]]))
fig.show(config=plotly_config)
fig.write_html("pwelch.html", config=plotly_config)

# %%
all_coh =  _load_xarray(base_folder/"rat_coh.zarr").sel(f=slice(3, 55))
for suffix in ["_1", "_2"]:
  all_coh["sig_group"+suffix] = xr.where(all_coh["sig_type"+suffix] == "eeg", "eeg", all_coh["structure"+suffix])
  all_coh["sig_group"+suffix] = xr.where(all_coh["neuron_type"+suffix] != "", all_coh["neuron_type"+suffix], all_coh["sig_group"+suffix])


coh_groups = ["condition", "sig_group_1", "sig_group_2", "sig_type_1", "sig_type_2", "is_APO", "has_swa"]
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
for st in ["bua", "neuron"]:
  plot_coh_power = np.abs(all_coh_grouped).sel(is_APO=False, has_swa=False, sig_type_1=[st, "eeg"], sig_type_2=[st])
  plot_coh_power = plot_coh_power.where(plot_coh_power.notnull(), drop=True)
  fig = line_error_bands(plot_coh_power.to_dataframe(name="coh").reset_index().dropna(subset="coh"), 
                x="f", y="coh", color="condition", facet_row="sig_group_2", facet_col="sig_group_1", error_y="sem",
                category_orders=dict(condition=[c for c in condition_order if c in plot_coh_power["condition"]], 
                                     sig_group_1=[sg for sg in sig_group_order if sg in plot_coh_power["sig_group_1"]],
                                     sig_group_2=[sg for sg in sig_group_order if sg in plot_coh_power["sig_group_2"]]),
                hover_data=["n_sigs","n_segments", "n_session_grp", "n_subjects", "sem"], title=f"{st.upper()} Coherence power for Rats", height=800)
  fig.show(config=plotly_config)
  fig.write_html(f"coh_power_{st}.html", config=plotly_config)

# %%
all_coh.sel(f=np.abs(all_coh).idxmax("f"))

# %%
for st in ["bua", "neuron"]:
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
  res = res.assign_coords({k: all_coh_grouped[k] for k in ["n_sigs", "n_subjects", "n_segments", "n_session_grp"]})
  res = res.sel(is_APO=False, has_swa=False, sig_type_1=[st, "eeg"], sig_type_2=[st, "eeg"], condition="Park")

  res= res.where(res.notnull(), drop=True)
  # display(res.notnull().groupby("sig_group_2").apply(lambda d: d.sum()))
  # display(res)
  f1 = px.line(res.to_dataframe(name="power").reset_index().dropna(), 
          y="power", x="angle", facet_row="sig_group_2", facet_col="sig_group_1", color="fmethod",
          category_orders=dict(sig_group_1=[sg for sg in sig_group_order if sg in res["sig_group_1"]],
                                      sig_group_2=[sg for sg in sig_group_order if sg in res["sig_group_2"]]),
          hover_data=["n_sigs","n_segments", "n_session_grp", "n_subjects"], height=800,
          title=f"{st.upper()} Coherence phase distribution for Rats")
  f1.write_html(f"coh_phase_distribution_{st}.html", config=plotly_config)
  display(f1)
  f2 = px.line(res.sel(sig_group_1="eeg", fmethod="20Hz").to_dataframe(name="power").reset_index().dropna(), 
          y="power", x="angle", color="sig_group_2",
          category_orders=dict(sig_group_1=[sg for sg in sig_group_order if sg in res["sig_group_1"]],
                                       sig_group_2=[sg for sg in sig_group_order if sg in res["sig_group_2"]]),
          hover_data=["n_sigs","n_segments", "n_session_grp", "n_subjects"],
          title=f"{st.upper()} Coherence phase distribution at 20Hz for Rats relative to EEG")
  f2.write_html(f"coh_phase_distribution_rEEG_{st}_20Hz.html", config=plotly_config)
  display(f2)

# %%
coh_phase_group_dict = {}
for name, ar in {"at20":all_coh.sel(f=20), "max(8, 35)":all_coh.sel(f=np.abs(all_coh).sel(f=slice(8, 35)).idxmax("f"))}.items():
  phase_coh = ar
  coh = xr.Dataset()
  coh["phase"] = xr.apply_ufunc(lambda a: np.angle(a, deg=True),phase_coh)
  coh["phase"] = xr.where(coh["phase"]<0, coh["phase"]+360, coh["phase"])
  coh["power"] = np.abs(phase_coh)
  
  tmp = []
  for _, g in coh["power"].groupby(coh_groups):
      r = (g-g.mean())/g.std()
      r = r-r.min()
      tmp.append(r)
  coh["power"] = xr.concat(tmp, dim="channel_pair")
  display(px.density_heatmap(
     coh.sel(channel_pair=(~coh["has_swa"]) & (~coh["is_APO"]) & (coh["condition"]=="Park") 
             & (coh["sig_type_1"].isin(["bua", "eeg"])& (coh["sig_type_2"].isin(["bua"])))).to_dataframe().reset_index(), 
     x="power", y="phase", facet_row="sig_group_1", facet_col="sig_group_2", histnorm="probability"))
  heatmap_groups = coh.groupby(dict(phase=xr.groupers.BinGrouper(np.linspace(0, 360, 13)), power=xr.groupers.BinGrouper(bins=8)) 
                              | {k: xr.groupers.UniqueGrouper() for k in coh_groups})
  coh_phase_groups = heatmap_groups.apply(lambda x: xr.DataArray(x.sizes["channel_pair"])).fillna(0).astype(int)
  coh_phase_groups["density"] = coh_phase_groups / coh_phase_groups.sum(["phase_bins", "power_bins"])
  coh_phase_groups["phase_bins_mid"] = xr.apply_ufunc(lambda x: x.mid,coh_phase_groups["phase_bins"], vectorize=True)
  coh_phase_groups["power_bins_right"] = xr.apply_ufunc(lambda x: x.right,coh_phase_groups["power_bins"], vectorize=True)
  coh_phase_groups = coh_phase_groups.assign_coords({k: all_coh_grouped[k] for k in ["n_sigs", "n_subjects", "n_segments", "n_session_grp"]})
  coh_phase_groups = coh_phase_groups.to_dataset(name="count").reset_coords(["density"])
  display(coh_phase_groups)
  coh_phase_group_dict[name] = coh_phase_groups




# %%
for st in ["bua", "neuron"]:
  for method, coh_phase_groups in coh_phase_group_dict.items():
    plot_coh_phase = coh_phase_groups.sel(is_APO=False, has_swa=False, sig_type_1=[st, "eeg"], sig_type_2=[st], condition="Park")
    plot_coh_phase = plot_coh_phase.stack(dim1=("sig_group_1", "sig_type_1"), dim2=("sig_group_2", "sig_type_2"), create_index=False)
    plot_coh_phase = plot_coh_phase.where(plot_coh_phase["n_sigs"]>5, drop=True)
    plot_coh_phase = plot_coh_phase.where((plot_coh_phase["count"] > 0).any(["phase_bins"]), drop=False).fillna(0)
    plot_coh_phase = plot_coh_phase.swap_dims(dim1="sig_group_1", dim2="sig_group_2", power_bins="power_bins_right", phase_bins="phase_bins_mid").set_coords("count")
    fig = faceted_imshow_xarray(plot_coh_phase["density"], r_dim="power_bins_right", theta_dim="phase_bins_mid", 
                                facet_row="sig_group_2",facet_col="sig_group_1", 
                                hover_data=["count", "n_sigs","n_segments", "n_session_grp", "n_subjects"],
                                subplot_width=320, subplot_height=250)
    fig.update_layout(title=f"{st.upper()} Coherence phase {method} for Rats")
    fig.show(config=plotly_config)
    # fig.write_html(f"coh_phase_{method}_{st}.html", config=plotly_config)



