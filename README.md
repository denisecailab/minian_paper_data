# Supplementary data for minian paper

This repository contains all the data and code used to generate the figures for the minian paper

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
* `validate.py` validate the spatial footprints and spike inference from two brain regions and generate fig 14.
* `validate_plc.py` validate spatial firing pattern of place cells across original and shuffled cell mappings and generate fig 15.
* `flow_chart.py` generate the flow chart of the pipeline (fig 1).

## data source

The folder `data/`, `minian_snapshot/`, and `ezTrack_snapshot/` are static and large, and hence not tracked with this git repo.
They are hosted as zip files on a google drive here: https://drive.google.com/drive/folders/1EpDquFInJRjtdBo_BMHsaFErmqr4w3lD?usp=sharing
