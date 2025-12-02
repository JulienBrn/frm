# %%
import plotly.express as px, plotly.graph_objects as go
from pathlib import Path
import xarray as xr, pandas as pd, numpy as np
from pipeline_helper import _load_xarray
from dask.diagnostics import ProgressBar
from dafn.plot_utilities import plotly_config, faceted_imshow_xarray, line_error_bands, filter_facet_categories
import itables, warnings
itables.init_notebook_mode(all_interactive=True)

# %%
def save_fig(fig: go.Figure, name: str, plot_type: str=None, sig_type: str=None, species: str=None, cond: str=None, baseline: str=None, fmethod: str=None):
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
    if fmethod is not None:
       path = path/fmethod.lower()
       title.append(fmethod.capitalize())
    if cond is not None:
       path = path/cond.lower()
       title.append(cond.capitalize())
    path = path/(name+".html")
    title = "--".join(title) + "--" + name
    fig.update_layout(title=title)
    fig.show(config=plotly_config)
    path.parent.mkdir(exist_ok=True, parents=True)
    fig.write_html(path, config=plotly_config)

def save_stats(arr: xr.DataArray, plot_type, subgroup):
    # display(arr)
    df = arr.to_dataset("stat").to_dataframe().reset_index()
    # display(df)
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
sig_group_order = ["eeg", "STR", "STN", "GPe", "Arky", "Proto", "STR<2Hz", "STR>2Hz"]
condition_order = ["Park", "CTL"]
species_order = ["Rat", "Monkey", "Human"]

# %%
all_welch =xr.open_dataarray(base_folder/"all_species_welch.zarr").sel(f=slice(3, 55)).compute().to_dataset(name="data")
all_coh = xr.open_dataarray(base_folder/"all_species_coh.zarr").sel(f=slice(3, 55)).compute().to_dataset(name="coh")
all_coh["data"] = np.abs(all_coh["coh"])**2
all_coh["data_angle"] = xr.apply_ufunc(np.angle, all_coh["coh"])
all_coh = all_coh.drop_vars("coh")
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
    pic_f = max_f.where((d["data"].sel(f=max_f) > d["data"].sel(f=max_f+1)) & (d["data"].sel(f=max_f) > d["data"].sel(f=max_f-1)))
    def max_of_grp(g: xr.DataArray):
       res = (g.sel(f=slice(8, 35)).sel({dim: g["condition"] == "Park"}).mean(dim) -  g.sel(f=slice(8, 35)).sel({dim: g["condition"] == "CTL"}).mean(dim)).idxmax("f")
       b = res.broadcast_like(g[dim])
       b = b.rename("best_f")
       return b
    all = []
    for _, grp in d["data"].groupby([c for c in grps if not c=="condition"]):
       all.append(max_of_grp(grp))
    d["max_f_grp"] = xr.concat(all, dim=dim)
    
    f_sel = xr.concat([max_f.assign_coords(fmethod="max"), 
                       pic_f.assign_coords(fmethod="pic"), 
                       xr.full_like(max_f, 20).assign_coords(fmethod="20Hz"), 
                       d["max_f_grp"].assign_coords(fmethod="max_f_grp")],
                       dim="fmethod")
    pow_at_f_sel = d["data"].sel(f=f_sel.fillna(8)).where(f_sel.notnull())
    # d["max_f"] = max_f
    auc = d["data"].sel(f=slice(8,35), drop=True).mean("f")
    metric = xr.concat([auc.assign_coords(fmethod="mean"), pow_at_f_sel.drop_vars("f")], dim="fmethod")
    # display(metric)
    def rel_condition(d):
      ctl = d.where(d["condition"]=="CTL", drop=True)
      ret = (d - ctl.mean(dim)) / ctl.std(dim)
      return ret
    #Because d["auc"].groupby([c for c in welch_groups if not c=="condition"]).map(rel_condition) does not work
    all = []
    for g, grp in metric.groupby([c for c in grps if not c=="condition"]):
       rc = rel_condition(grp)
       all.append(rc)

    pow_metric = xr.concat([xr.concat(all, dim=dim).assign_coords(cond_norm="zscored"), metric.assign_coords(cond_norm="none")], dim="cond_norm")
    d["pow_metric"] = pow_metric
    d["f_sel"] = f_sel

    
display(all_coh)

# %%

tmp_coh = all_coh.set_coords([d for d in all_coh.data_vars if not "f" in all_coh[d].dims])
phase_coh = tmp_coh.sel(f=tmp_coh["f_sel"].fillna(8)).where(tmp_coh["f_sel"].notnull())
# xr.concat([tmp_coh.sel(f=20).assign_coords(fsel_method="20Hz"), 
          #  tmp_coh.sel(f=all_coh["max_f"].fillna(8)).where(all_coh["max_f"].notnull()).assign_coords(fsel_method="maxf")], dim="fsel_method")
phase_coh["angle_rad"] = xr.DataArray(np.linspace(-np.pi, np.pi, 100, endpoint=False), dims="angle")
phase_coh["angle"] = phase_coh["angle_rad"] * 180/np.pi
def angle_diff(a, b):
    return (a - b + np.pi) % (2 * np.pi) - np.pi
phase_coh["data"] = phase_coh["data"]*np.exp(-10*(angle_diff(phase_coh["angle_rad"], phase_coh["data_angle"]))**2)
phase_coh["data"] = phase_coh["data"]/phase_coh["data"].sum("angle")
phase_coh = phase_coh[["data"]]
phase_coh

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
  def stat_group(d: xr.DataArray):
    if d.size == 0:
      display(d)
      raise Exception("Problem")
      
    def stat_test(data: np.ndarray, col, v1, v2, fn):
      arr1 = data[col==v1]
      arr2 = data[col==v2]
      arr1 = arr1[~np.isnan(arr1)]
      arr2 = arr2[~np.isnan(arr2)]
      if arr1.size ==0 or arr2.size==0:
          stat = np.nan
      else: 
        stat = fn(arr1, arr2).pvalue
      with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m1 = np.median(arr1) 
        m2 = np.median(arr2)
        mad1 = np.median(np.abs(arr1-m1))
        mad2 = np.median(np.abs(arr2-m2))
        return np.array([stat, arr1.size, arr2.size, arr1.mean(), arr2.mean(), arr1.std(), arr2.std(), m1, m2, mad1, mad2])
    try:
      res =  xr.apply_ufunc(stat_test, d, d[col], all_stats[col+"_pair_1"], all_stats[col+"_pair_2"], stat_funcs, input_core_dims=[[dim], [dim], [] ,[], []], 
                      output_core_dims=[["stat"]], vectorize=True, output_dtypes=[float])
    except Exception:
      display(d)
      display(all_stats)
      display(col)
      raise
    res["stat"] = np.array(["pvalue", "n1", "n2", "avg1", "avg2", "std1", "std2", "med1", "med2", "mad1", "mad2"])
    return res
  for col in compare_cols:
    if all_stats[col+"_pair_1"].size ==0:
      continue
    ks_stats = d[value].groupby([c for c in groups if not c==col]).apply(stat_group)
    all_stats["stat_cmp_"+col] = ks_stats
  return all_stats

for v in "pow_metric", "f_sel":
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
    if "f" in d["data"].dims:
      for b in d["baseline"].to_numpy():
          tmp = []
          for fm in d["fmethod"].to_numpy():
            if fm=="mean":
               continue
            tmp.append(d.sel(baseline=b, fmethod=fm).groupby(groups+ ["f_sel"]).apply(lambda d: d["f_sel"].count()).fillna(0))
          all.append(xr.concat(tmp, dim="fmethod"))
      all = xr.concat(all, dim="baseline")
      res["n_f_sel"] = all
      res["n_f_sel"] = res["n_f_sel"].where(res["n_f_sel"].sum("f_sel")>0)
      res["%_f_sel"] = res["n_f_sel"]/res["n_f_sel"].sum("f_sel")
      # res["f_sel_"+name] = d.set_coords("f_sel")["max_f_pow"].groupby(groups+ ["max_f"]).mean()
    return res

welch_grouped = compute_groups(all_welch, welch_groups, "welch")
coh_grouped = compute_groups(all_coh, coh_groups, "coh")
phase_coh_grouped = compute_groups(phase_coh, coh_groups, "coh_phase")
display(welch_grouped)
display(coh_grouped)
display(phase_coh_grouped)

# %%
d_coh = coh_grouped.sel(sig_type_1="eeg").rename(sig_group_2="sig_group", sig_type_2="sig_type")
d_phase = phase_coh_grouped.sel(sig_type_1="eeg").rename(sig_group_2="sig_group", sig_type_2="sig_type")

def plot(data, name, **kwargs):
  func = line_error_bands if "error_y" in kwargs else px.line
  data_df = xr_to_pd(data, name)
  return filter_facet_categories(func)(data_df, **kwargs,
                    hover_data=[c for c in ["n_sigs", "n_subjects", "sem", "n_f_sel"] if c in data_df.columns], 
                    category_orders=dict(condition=condition_order, sig_group=sig_group_order, species=species_order),
                    subplot_height=200, subplot_width=400,
  )


for name, d in ("coh", d_coh), ("welch", welch_grouped), ("coh_phase", d_phase):
  for b in d["baseline"].to_numpy():
    for st in d["sig_type"].to_numpy():
      if st=="eeg":
        continue
      selection = "" if name=="welch" else " rel EEG"
      ytype = "Power Spectrum" if not "phase" in name else "Phase Proba"
      x_col = "f" if not "phase" in name else "angle"
      fmethod = [None] if not "phase" in name else d["fmethod"].to_numpy()
      for fsel in fmethod:
        plot_d = d.sel(sig_type=[st], baseline=b) if fsel is None else d.sel(sig_type=[st], baseline=b, fmethod=fsel)

        fig = plot(plot_d, name, x=x_col, y=name, color="condition", facet_row="sig_group", facet_col="species", error_y="sem")
        save_fig(fig, name=f"{ytype} Condition"+selection, plot_type=name, sig_type=st, baseline=b, fmethod=fsel)

        fig = plot(plot_d.sel(condition="Park"), name, 
                  x=x_col, y=name, color="species", facet_row="sig_group", facet_col="condition", error_y="sem")
        save_fig(fig, name=f"{ytype} Species"+selection, plot_type=name, sig_type=st, baseline=b, cond="Park", fmethod=fsel)

        fig = plot(plot_d.sel(condition="Park"), name, 
                  x=x_col, y=name, color="sig_group", facet_row="species", facet_col="condition", error_y="sem")
        save_fig(fig, name=f"{ytype} Structure"+selection, plot_type=name, sig_type=st, baseline=b, cond="Park", fmethod=fsel)

      if "f" in d.dims:
        for fsel in d["fmethod"].to_numpy():
          if fsel in ["mean", "20Hz"]:
            continue
          plot_d = d.sel(sig_type=[st], baseline=b, fmethod=fsel)
          fig = plot(plot_d.assign(max_f_centered=plot_d["f_sel"]-0.5), "%_f_sel", 
                    x="max_f_centered", y="%_f_sel", color="species", facet_row="condition", facet_col="sig_group",
                    line_shape='hv',)
          save_fig(fig, name="max_f dist"+selection, plot_type=name, sig_type=st, baseline=b, fmethod=fsel)

        # fig = plot(d.sel(sig_type=[st, "eeg"], baseline=b), "max_f_"+name, 
        #               x="max_f", y="max_f_"+name, color="species", facet_row="condition", facet_col="sig_group")
        # save_fig(fig, name="max_f pow"+selection, plot_type=name, sig_type=st, baseline=b)


