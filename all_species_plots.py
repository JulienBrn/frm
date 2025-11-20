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
base_folder = Path("/media/julienb/T7 Shield/Revue-FRM/AnalysisData8/")
result_folder = Path("./all_species")
result_folder.mkdir(exist_ok=True, parents=True)
all_welch =_load_xarray(base_folder/"all_species_welch.zarr").sel(f=slice(3, 55))
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
for st in ["bua", "neuron"]:
  plot_welch = all_welch_grouped.sel(sig_type=[st])
  plot_welch = plot_welch.where(plot_welch.notnull(), drop=True)

  fig = line_error_bands(plot_welch.to_dataframe(name="welch").reset_index(), 
                x="f", y="welch", color="condition", facet_row="sig_group", facet_col="species", error_y="sem",
                hover_data=["n_sigs", "n_subjects", "sem"], height=800, title=f"{st.upper()} Pwelch",
                category_orders=dict(condition=condition_order, sig_group=[sg for sg in sig_group_order if sg in plot_welch["sig_group"]]))
  fig.show(config=plotly_config)
  fig.write_html(result_folder/f"pwelch_compare_cond{st}.html", config=plotly_config)

  fig = line_error_bands(plot_welch.sel(condition="Park").to_dataframe(name="welch").reset_index(), 
                x="f", y="welch", color="species", facet_row="condition", facet_col="sig_group", error_y="sem",
                hover_data=["n_sigs", "n_subjects", "sem"], height=400, title=f"{st.upper()} Pwelch Park",
                category_orders=dict(sig_group=[sg for sg in sig_group_order if sg in plot_welch["sig_group"]]))
  fig.show(config=plotly_config)
  fig.write_html(result_folder/f"pwelch_compare_species{st}.html", config=plotly_config)

  fig = line_error_bands(plot_welch.sel(condition="Park").to_dataframe(name="welch").reset_index(), 
                x="f", y="welch", color="sig_group", facet_row="condition", facet_col="species", error_y="sem",
                hover_data=["n_sigs", "n_subjects", "sem"], height=400, title=f"{st.upper()} Pwelch Park",
                category_orders=dict(csig_group=[sg for sg in sig_group_order if sg in plot_welch["sig_group"]]))
  fig.show(config=plotly_config)
  fig.write_html(result_folder/f"pwelch_compare_structure{st}.html", config=plotly_config)


