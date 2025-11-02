# Binomial slices — README

A short guide to set up the environment and generate statistics CSV files from histogram files using the script `Python_Downstream/calc_binom_slices.py`.

Inteneded to run on rcf which has access to dylan's local python installation. Also assumes using bash shell.

## Quick start

1. If the repository has not been cloned yet: make `clone_and_source.sh` executable and source it (this will clone the repo and set up environment variables / python path as the script is designed).

Make the script executable and run it in the current shell so any exported variables take effect.

```bash
# Make clone_and_source.sh executable and source it (runs inside current shell)
chmod +x clone_and_source.sh
source ./clone_and_source.sh
```

2. If the repo is already cloned: make `source_python.sh` executable and source it to set up the Python environment / paths.

`source_python.sh` configures environment for running the Python scripts.

```bash
# If repo already exists, just source the Python environment script
chmod +x source_python.sh
source ./source_python.sh
```

## Configure `calc_binom_slices.py`

Open `Python_Downstream/calc_binom_slices.py` and edit `init_pars()` to point at your histogram files and desired outputs:

- Set `base_path` to the directory containing the histogram folders (the script expects a `Data` and a `Data_Mix` directory).
- Set `csv_path` to the output CSV filename where statistics will be written.
- Adjust other options in `init_pars()` such as:
  - `threads` (number of worker processes)
  - `stats` (which statistics are calculated; configured via `define_stats`)
  - `min_events`, `min_bs`, `out_bs`, `save_cent`, `save_data_type`, `save_stat`, `diff`, etc.
- Select which datasets to process by editing `define_datasets()` — add or modify dataset dictionary entries to match your histogram folder naming and keys.

## Run the statistics generator

Run the script with Python 3 from the repository root (the `source_*.sh` scripts should have set up the environment).

```bash
# Run the statistics extraction script
python3 Python_Downstream/calc_binom_slices.py
```

The script will:
- read histogram `.txt` files under `base_path` (and the `_Mix` sibling path for mixed data),
- compute the configured statistics per dataset / division / centrality / total_protons,
- write the aggregated results to the `csv_path` CSV file.

## Notes and tips
- Use `pars['only_new'] = True` and `pars['csv_append'] = True` in `init_pars()` to append only new datasets to an existing CSV.
- Use `pars['check_only'] = True` to validate all files can be read without doing the full computation.
- Ensure `info.txt` files exist in each raw data directory; `get_min_counts()` reads `info.txt` to determine binning/resample details.
- If you see missing mixed files, the script prints `Mixed file missing!` messages and continues.

## Troubleshooting
- If permission errors: ensure the shell scripts are executable (`chmod +x`).
- If the script produces no rows, verify `base_path` points to the correct directory structure and `define_datasets()` selects the expected folders.

