# %%
import plotly.express as px, plotly.graph_objects as go
from pathlib import Path
import xarray as xr, pandas as pd, numpy as np
from pipeline_helper import _load_xarray
from dask.diagnostics import ProgressBar
from dafn.plot_utilities import plotly_config, faceted_imshow_xarray, line_error_bands, filter_facet_categories
import itables
itables.init_notebook_mode(all_interactive=True)

# %%
def save_fig(fig: go.Figure, name: str, plot_type: str=None, sig_type: str=None, species: str=None, cond: str=None, baseline: str=None):
    path = Path("./results/figures")
    title = []
    if plot_type is not None:
       path = path/plot_type.lower()
       title.append(plot_type.capitalize())
    if sig_type is not None:
       path = path/sig_type.lower()
       title.append(sig_type.upper())
    if baseline is not None:
       path = path/baseline.lower()
       title.append(baseline.capitalize())
    if species is not None:
       path = path/species.lower()
       title.append(species.capitalize())
    if cond is not None:
       path = path/cond.lower()
       title.append(cond.capitalize())
    path = path/(name+".html")
    title = "--".join(title) + "--" + name
    fig.update_layout(title=title)
    fig.show()
    path.parent.mkdir(exist_ok=True, parents=True)
    fig.write_html(path)

def save_stats(arr: xr.DataArray, plot_type, subgroup):
    df = arr.to_dataframe(name="pvalue").reset_index()
    empty_dims = [c for c in arr.dims if not c in arr.coords]
    df= df.drop(columns=empty_dims)
    df = df.loc[df["pvalue"].notna()]
    path = Path("./results/stats")/plot_type/subgroup/(arr.name+".xlsx")
    path.parent.mkdir(exist_ok=True, parents=True)
    df.to_excel(path, index=False)

def xr_to_pd(ds:xr.Dataset, col):
  ds = ds.drop_dims([d for d in ds.dims if not d in ds[col].dims])
  df = ds.to_dataframe().reset_index()
  df=df.loc[df[col].notna()]
  return df

base_folder = Path("/media/julienb/T7 Shield/Revue-FRM/AnalysisData8/")

# %%
sig_group_order = ["eeg", "STR", "STN", "GPe", "Arky", "Proto"]
condition_order = ["Park", "CTL"]
species_order = ["Rat", "Monkey", "Human"]

# %%
all_welch =xr.open_dataarray(base_folder/"all_species_welch.zarr").sel(f=slice(3, 55)).compute().to_dataset(name="data")
all_coh = xr.open_dataarray(base_folder/"all_species_coh.zarr").sel(f=slice(3, 55)).compute().to_dataset(name="coh")
all_coh["data"] = np.abs(all_coh["coh"])**2
display(all_welch)
display(all_coh)


# %%
def assign_grp(d: xr.Dataset, suffix):
   d=d.assign_coords({"sig_group"+suffix: 
                      xr.where(d["sig_type"+suffix] == "eeg", "eeg", 
                               xr.where(d["neuron_type"+suffix]!="", d["neuron_type"+suffix], d["structure"+suffix])
                               )
                      })
   return d
all_welch["channel"] = np.arange(all_welch.sizes["channel"])
all_coh["channel_pair"] = np.arange(all_coh.sizes["channel_pair"])
all_welch = assign_grp(all_welch, "")
all_coh = assign_grp(assign_grp(all_coh, "_1"), "_2")



welch_groups = ["species", "condition", "sig_group", "sig_type"]
coh_groups = ["species", "condition", "sig_group_1", "sig_group_2", "sig_type_1", "sig_type_2"]
welch_compare_cols = ["condition", "species", "sig_group"]
coh_compare_cols = ["condition", "species"]
from scipy.stats import ks_2samp, mannwhitneyu
stats_funcs=xr.DataArray([ks_2samp, mannwhitneyu], dims="stat_fn")
stats_funcs["stat_fn"] = ["ks", "mannwhitneyu"]

display(all_coh)
display(all_welch)

# %%



for d, grps in (all_welch, welch_groups), (all_coh, coh_groups):
    rel_data = d["data"] - ((d["f"]-8) * (d["data"].sel(f=35)-d["data"].sel(f=8))/(35-8) + d["data"].sel(f=8))
    d["data"] = xr.concat([d["data"].assign_coords(baseline="zero"), rel_data.assign_coords(baseline="interp(8, 35)")], dim="baseline")
    dim =d["condition"].dims[0]
    max_f = d["data"].sel(f=slice(8, 35)).idxmax("f")
    max_f = max_f.where((d["data"].sel(f=max_f) > d["data"].sel(f=max_f+1)) & (d["data"].sel(f=max_f) > d["data"].sel(f=max_f-1)))
    d["max_f"] = max_f
    auc = d["data"].sel(f=slice(8,35)).mean("f")
    def rel_condition(d):
      ctl = d.where(d["condition"]=="CTL", drop=True)
      ret = (d - ctl.mean(dim)) / ctl.std(dim)
      return ret
    #Because d["auc"].groupby([c for c in welch_groups if not c=="condition"]).map(rel_condition) does not work
    all = []
    for g, grp in auc.groupby([c for c in grps if not c=="condition"]):
       rc = rel_condition(grp)
       all.append(rc)

    
    d["auc"] = xr.concat([xr.concat(all, dim=dim).assign_coords(cond_norm="zscored"), auc.assign_coords(cond_norm="none")], dim="cond_norm")
    d["max_f_pow"] = d["data"].sel(f=d["max_f"].fillna(8)).where(d["max_f"])
    
display(all_welch)

# %%

def compute_stats(d: xr.Dataset, value, groups, compare_cols, stat_funcs):
  dim =d["condition"].dims[0]
  all_stats = xr.Dataset()
  for col in compare_cols:
      unique_values = np.unique(d[col].to_numpy())
      all_stats[col+"_pair_1"] = xr.DataArray([v1 for v1 in unique_values for v2 in unique_values], dims=col+"_pair")
      all_stats[col+"_pair_2"] = xr.DataArray([v2 for v1 in unique_values for v2 in unique_values], dims=col+"_pair")
      all_stats = all_stats.set_coords([col+"_pair_1", col+"_pair_2"])
      all_stats = all_stats.sel({col+"_pair": all_stats[col+"_pair_1"] < all_stats[col+"_pair_2"]})

  def stat_test(data, col, v1, v2, fn):
    
    arr1 = data[col==v1]
    arr2 = data[col==v2]
    if arr1.size ==0 or arr2.size==0:
        return np.nan
    return fn(arr1, arr2).pvalue

  for col in compare_cols:
    ks_stats = d[value].groupby([c for c in groups if not c==col]).apply(lambda d:
                xr.apply_ufunc(stat_test, d, d[col], all_stats[col+"_pair_1"], all_stats[col+"_pair_2"], stat_funcs, input_core_dims=[[dim], [dim], [] ,[], []], vectorize=True, output_dtypes=[float]))
    all_stats["stat_cmp_"+col] = ks_stats
  return all_stats

for v in "auc", "max_f":
  for n, d, grps, comp in ("welch", all_welch, welch_groups, welch_compare_cols), ("coh", all_coh, coh_groups, coh_compare_cols):
    stats = compute_stats(d, v, grps, comp, stats_funcs)
    for arr in stats.data_vars:
      save_stats(stats[arr], n, v)



# %%
def compute_groups(d: xr.Dataset, groups: list, name):
    dim = d[groups[0]].dims[0]
    grp = d["data"].groupby(groups)
    res = xr.Dataset()
    res[name] = grp.mean()
    # res["auc_"+name] = d["auc"].groupby(groups).mean()
    res["n_sigs"] =  grp.map(lambda x: xr.DataArray(x.sizes[dim])).fillna(0).astype(int)
    res["n_subjects"] = grp.apply(lambda x: xr.DataArray(len(np.unique(x["subject"])))).fillna(0).astype(int)
    res["sem"] = grp.std() / np.sqrt(res["n_sigs"])
    # res["n_max_f"] = d.groupby(groups+ ["max_f"]).apply(lambda d: xr.DataArray(d.sizes[dim])).fillna(0)
    all = []
    for b in d["baseline"].to_numpy():
        all.append(d.sel(baseline=b).groupby(groups+ ["max_f"]).apply(lambda d: xr.DataArray(d.sizes[dim])).fillna(0))
    res["n_max_f"] = xr.concat(all, dim="baseline")
    res["n_max_f"] = res["n_max_f"].where(res["n_max_f"].sum("max_f")>0)
    res["%_max_f"] = res["n_max_f"]/res["n_max_f"].sum("max_f")
    res["max_f_"+name] = d.set_coords("max_f")["max_f_pow"].groupby(groups+ ["max_f"]).mean()
    return res

welch_grouped = compute_groups(all_welch, welch_groups, "welch")
coh_grouped = compute_groups(all_coh, coh_groups, "coh")
display(welch_grouped)
display(coh_grouped)

# %%
d_coh = coh_grouped.sel(sig_type_1="eeg").rename(sig_group_2="sig_group", sig_type_2="sig_type")
for name, d in ("coh", d_coh), ("welch", welch_grouped):
  for b in d["baseline"].to_numpy():
    for st in d["sig_type"].to_numpy():
      if st=="eeg":
        continue
      selection = "" if name=="welch" else " rel EEG"
      fig = filter_facet_categories(line_error_bands)(xr_to_pd(d.sel(sig_type=[st], baseline=b), name), 
                    x="f", y=name, color="condition", facet_row="sig_group", facet_col="species", error_y="sem",
                    hover_data=["n_sigs", "n_subjects", "sem"], 
                    category_orders=dict(condition=condition_order, sig_group=sig_group_order),
                    subplot_height=200, subplot_width=400,
                    )
      save_fig(fig, name="Power Spectrum"+selection, plot_type=name, sig_type=st, baseline=b)
      fig = filter_facet_categories(line_error_bands)(xr_to_pd(d.sel(sig_type=[st], baseline=b, condition="Park"), name), 
                    x="f", y=name, color="species", facet_row="sig_group", facet_col="condition", error_y="sem",
                    hover_data=["n_sigs", "n_subjects", "sem"], 
                    category_orders=dict(condition=condition_order, sig_group=sig_group_order),
                    subplot_height=200, subplot_width=400,
                    )
      save_fig(fig, name="Power Spectrum Species"+selection, plot_type=name, sig_type=st, baseline=b, cond="Park")
      fig = filter_facet_categories(line_error_bands)(xr_to_pd(d.sel(sig_type=[st], baseline=b, condition="Park"), name), 
                    x="f", y=name, color="sig_group", facet_row="species", facet_col="condition", error_y="sem",
                    hover_data=["n_sigs", "n_subjects", "sem"], 
                    category_orders=dict(condition=condition_order, sig_group=sig_group_order),
                    subplot_height=200, subplot_width=400,
                    )
      save_fig(fig, name="Power Spectrum Structure"+selection, plot_type=name, sig_type=st, baseline=b, cond="Park")
      fig = filter_facet_categories(px.line)(xr_to_pd(d.assign(max_f_centered=d["max_f"]-0.5).sel(sig_type=[st, "eeg"], baseline=b), "%_max_f"), 
                    x="max_f_centered", y="%_max_f", color="species", facet_row="condition", facet_col="sig_group",
                    line_shape='hv',
                    hover_data=["max_f", "n_max_f", "n_sigs", "n_subjects"],height=800, 
                    category_orders=dict(condition=condition_order, sig_group=sig_group_order, species=species_order),
                    subplot_height=200, subplot_width=400,
                    )
      save_fig(fig, name="max_f dist"+selection, plot_type=name, sig_type=st, baseline=b)
      fig = filter_facet_categories(px.line)(xr_to_pd(d.sel(sig_type=[st, "eeg"], baseline=b), "max_f_"+name), 
                    x="max_f", y="max_f_"+name, color="species", facet_row="condition", facet_col="sig_group",
                    hover_data=["max_f", "n_max_f", "n_sigs", "n_subjects"],height=800, 
                    category_orders=dict(condition=condition_order, sig_group=sig_group_order, species=species_order),
                    subplot_height=200, subplot_width=400,
                    )
      save_fig(fig, name="max_f pow"+selection, plot_type=name, sig_type=st, baseline=b)

# %%
for b in coh_grouped["baseline"].to_numpy():
  for st in coh_grouped["sig_type_1"].to_numpy():
    fig = filter_facet_categories(line_error_bands)(xr_to_pd(coh_grouped.sel(sig_type_1="eeg", sig_type_2=st, baseline=b), "welch"), 
                  x="f", y="coh", color="condition", facet_row="sig_group_2", facet_col="species", error_y="sem",
                  hover_data=["n_sigs", "n_subjects", "sem"], 
                  category_orders=dict(condition=condition_order, sig_group=sig_group_order),
                  subplot_height=200, 
                  )
    save_fig(fig, name="Power Spectrum", plot_type="welch", sig_type=st, baseline=b)
    fig = filter_facet_categories(px.line)(xr_to_pd(welch_grouped.assign(max_f_centered=welch_grouped["max_f"]-0.5).sel(sig_type=st, baseline=b), "%_max_f"), 
                  x="max_f_centered", y="%_max_f", color="species", facet_row="condition", facet_col="sig_group",
                  line_shape='hv',
                  hover_data=["max_f", "n_max_f", "n_sigs", "n_subjects"],height=800, 
                  category_orders=dict(condition=condition_order, sig_group=sig_group_order, species=species_order),
                  subplot_height=200, 
                  )
    save_fig(fig, name="max_f dist", plot_type="welch", sig_type=st, baseline=b)
    fig = filter_facet_categories(px.line)(xr_to_pd(welch_grouped.sel(sig_type=st, baseline=b), "max_f_welch"), 
                  x="max_f", y="max_f_welch", color="species", facet_row="condition", facet_col="sig_group",
                  hover_data=["max_f", "n_max_f", "n_sigs", "n_subjects"],height=800, 
                  category_orders=dict(condition=condition_order, sig_group=sig_group_order, species=species_order),
                  subplot_height=200, 
                  )
    save_fig(fig, name="max_f pow", plot_type="welch", sig_type=st, baseline=b)

# %%
raise Exception("stop")

# %%


all_welch =_load_xarray(base_folder/"all_species_welch.zarr").sel(f=slice(3, 55))
# all_welch =xr.open_dataarray(base_folder/"all_species_welch.h5").sel(f=slice(3, 55))

max_f = all_welch.sel(f=slice(8, 35)).idxmax("f")
max_f = max_f.where((all_welch.sel(f=max_f) > all_welch.sel(f=max_f+1)) & (all_welch.sel(f=max_f) > all_welch.sel(f=max_f-1)))
all_welch["max_f"] = max_f
all_welch

# %%
welch_groups = ["condition", "sig_group", "sig_type", "species"]
welch_group = all_welch.to_dataset(name="welch").groupby(welch_groups)
all_welch_grouped = welch_group.mean()["welch"]
all_welch_grouped["n_sigs"] = welch_group.map(lambda x: xr.DataArray(x.sizes["channel"])).fillna(0).astype(int)
all_welch_grouped["n_subjects"] = welch_group.apply(lambda x: xr.DataArray(len(np.unique(x["subject"])))).fillna(0).astype(int)
all_welch_grouped["sem"] = welch_group.std()["welch"] / np.sqrt(all_welch_grouped["n_sigs"])
all_welch_grouped = all_welch_grouped.to_dataset(name="welch")
all_welch_grouped["n_max_f"] = all_welch.groupby(welch_groups+ ["max_f"]).apply(lambda d: xr.DataArray(d.sizes["channel"])).fillna(0)
display(all_welch_grouped)

# %%

# for st in ["eeg", "lfp", "bua", "neuron"]:
#   plot_max = max_f.sel(channel=(max_f["sig_type"]==st) & (max_f["condition"]=="Park"))
#   fig = px.histogram(plot_max.to_dataframe(name="f").reset_index(), x="f", facet_row="species", facet_col="sig_group",
#                      barmode="group", histnorm="probability", title=f"{st.upper()} Park Max f distribution")
#   if show:
#     display(fig)
#   (welch_folder/st).mkdir(exist_ok=True, parents=True)
#   fig.write_html(welch_folder/st/f"max_f_park.html", config=plotly_config)

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
  (welch_folder/st).mkdir(exist_ok=True, parents=True)
  fig.write_html(welch_folder/st/f"pwelch_compare_cond.html", config=plotly_config)

  fig = line_error_bands(plot_welch.sel(condition="Park").to_dataframe(name="welch").reset_index(), 
                x="f", y="welch", color="species", facet_row="condition", facet_col="sig_group", error_y="sem",
                hover_data=["n_sigs", "n_subjects", "sem"], height=400, title=f"{st.upper()} Pwelch Park",
                category_orders=dict(sig_group=[sg for sg in sig_group_order if sg in plot_welch["sig_group"]]))
  if show:
    fig.show(config=plotly_config)
  fig.write_html(welch_folder/st/f"pwelch_compare_species.html", config=plotly_config)

  fig = line_error_bands(plot_welch.sel(condition="Park").to_dataframe(name="welch").reset_index(), 
                x="f", y="welch", color="sig_group", facet_row="condition", facet_col="species", error_y="sem",
                hover_data=["n_sigs", "n_subjects", "sem"], height=400, title=f"{st.upper()} Pwelch Park",
                category_orders=dict(csig_group=[sg for sg in sig_group_order if sg in plot_welch["sig_group"]]))
  if show:
    fig.show(config=plotly_config)
  fig.write_html(welch_folder/st/f"pwelch_compare_structure.html", config=plotly_config)

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
max_f = all_coh.sel(f=slice(8, 35)).idxmax("f")
max_f = max_f.where((all_coh.sel(f=max_f) > all_coh.sel(f=max_f+1)) & (all_coh.sel(f=max_f) > all_coh.sel(f=max_f-1)))
for species in all_coh_grouped["species"].to_numpy():
  for st in ["eeg", "lfp", "bua", "neuron"]:
    plot_max = max_f.sel(channel_pair=(max_f["sig_type_1"]==st) & (max_f["sig_type_2"]==st) & (max_f["condition"]=="Park"))
    fig = px.histogram(plot_max.to_dataframe(name="f").reset_index(), x="f", facet_row="sig_group_1", facet_col="sig_group_2",
                      barmode="group", histnorm="probability", title=f"{st.upper()} Park Max f distribution", height=800)
    if show:
      display(fig)
    (coh_folder/species/st).mkdir(exist_ok=True, parents=True)
    fig.write_html(coh_folder/species/st/f"max_f_park.html", config=plotly_config)

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
    (coh_folder/species/st/"detailed").mkdir(exist_ok=True, parents=True)
    fig.write_html(coh_folder/species/st/f"detailed/coh_power.html", config=plotly_config)

    subplot = plot_coh_power.sel(species=species, sig_group_2="eeg", condition="Park", sig_type_1=[st])
    fig = line_error_bands(subplot.to_dataframe(name="coh").reset_index().dropna(subset="coh"), 
                  x="f", y="coh", color="sig_group_1", facet_row="sig_group_2", error_y="sem",
                  category_orders=dict(condition=[c for c in condition_order if c in subplot["condition"]], 
                                      sig_group_1=[sg for sg in sig_group_order if sg in subplot["sig_group_1"]],
                                      sig_group_2=[sg for sg in sig_group_order if sg in subplot["sig_group_2"]]),
                  hover_data=["n_sigs", "n_subjects", "sem"], title=f"{st.upper()} Coherence power for {species} Park rel EEG", height=400, width=600)
    if show:
      fig.show(config=plotly_config)
    fig.write_html(coh_folder/species/st/f"eeg_park_coh_power.html", config=plotly_config)

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
    f1.write_html(coh_folder/species/st/f"detailed/coh_phase_distribution.html", config=plotly_config)
    if show:
      display(f1)
    f2 = px.line(res.sel(sig_group_1="eeg", species=species).to_dataframe(name="power").reset_index().dropna(), 
            y="power", x="angle", color="sig_group_2", facet_col = "fmethod",
            category_orders=dict(sig_group_1=[sg for sg in sig_group_order if sg in res["sig_group_1"]],
                                        sig_group_2=[sg for sg in sig_group_order if sg in res["sig_group_2"]]),
            hover_data=["n_sigs","n_subjects"],
            title=f"{st.upper()} Coherence phase distribution at 20Hz for {species} relative to EEG")
    f2.write_html(coh_folder/species/st/f"coh_phase_distribution_relEEG.html", config=plotly_config)
    if show:
      display(f2)


