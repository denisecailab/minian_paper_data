# Supplementary data for minian paper

This repository contains the data and code used to generate Figure 1-15 and Figure 20 of the manuscript "Minian an Open-source Miniscope Analysis Pipeline"

## repo content

* `data/` contains raw and intermediate data.
* `minian_snapshot/` contains a copy of minian that's used for analysis.
* `ezTrack_snapshot/` contains a copy of [ezTrack](https://github.com/denisecailab/ezTrack) that's used for analysis.
* `figs/` contains all the generated figures.
* `misc/` contains some simple script to generate the dependency table used in the paper.
* `concat_video.ipynb` concatenate behavior videos so they can be easily processed with [ezTrack](https://github.com/denisecailab/ezTrack).
* `cross-registration.ipynb` is a modified version of minian notebook that register cells across sessions.
* `cross-registration_shuffle.ipynb` is a modified version of minian notebook that shuffle the location of cells (spatial footprints) and generated cell mappings across sessions.
* `pipeline_paper.ipynb` is a modified version of minian pipeline to facilitate taking snapshot of visualizations.
* `place_cell.ipynb` / `place_cell.py` extracts place cells from neural activities based on critirias.
* `validate.py` generate a validation plot that is now obsolete.
* `validate_plc.py` validate spatial firing pattern of place cells across original and shuffled cell mappings and generate fig 20.
* `flow_chart.py` generate the flow chart of the pipeline (fig 1).

## steps to reproduce

### create environments

1. `conda env create -n minian_paper_data -f misc/encironment.yml`

### get data

The source data are hosted on [figshare](https://doi.org/10.6084/m9.figshare.c.5987038.v1).
We have a convenient script to retrieve all the necessary data.

1. `python get_data.py`

### reproduce figure 1

1. `python flow_chart.py`

### reproduce figure 2-15

Run through `pipeline_paper.ipynb` and use the developer tool in the browser (preferrable google chrome) to take snapshot with a specific size.

### reproduce figure 20

#### rerun minian and ezTrack processing (Optional)

Since all the intermediate data are retrieved with `get_data.py` under `./data/inter`, this section can be skipped if you just want to reproduce plotting of Figure 20.

1. run through `./concat_video.ipynb` and then `./ezTrack_snapshot/LocationTracking_BatchProcess.ipynb` to produce location tracking results.
2. for each `pipeline.ipynb` under `./data/pfd2`, change `minian_path` to `"../../../minian_snapshot"` and run through the notebook to reproduce minian results.
3. run through `./cross-registration.ipynb` and `./cross-registration_shuffle.ipynb` to reproduce cross registrationi results.
4. run through `./place_cell.ipynb` to classify place cells.

#### reproduce figure 20

1. `python validate_plc.py`

