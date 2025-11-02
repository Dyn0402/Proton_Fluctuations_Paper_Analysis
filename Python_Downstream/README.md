# QGP_Scripts

This is a catch-all repository for scripts related to Quark-Gluon Plasma (QGP) research at the STAR experiment. 
It includes various scripts for data analysis, simulation, and visualization. 
The focus is on the analysis of proton clustering in azimuthal partitions, though 
other analyses generally related to the STAR experiment and moment analyeses in particular are also included.


## Proton Clustering in Azimuthal Partitions Analysis

This downstream python code runs on histogram text files produced from TreeReader in the 
[QGP_Fluctuations](https://github.com/Dyn0402/QGP_Fluctuations) repository.

Only two scripts are needed to run the analysis:
- `calc_binom_slices.py`: This script takes as input the histogram text files produced by TreeReader in 
[QGP_Fluctuations](https://github.com/Dyn0402/QGP_Fluctuations) and outputs a csv file containing the moments 
of these histograms. To run, all run parameters are localized to a `pars` dictionary in the `init_pars` function 
and the objects in the `define_datasets` function:
  - In `init_pars`, change the `base_path` to the directory containing `Data` and `Data_Mix` directories, 
  - which house the histogram text files from TreeReader.
  - Change any other parameters in `init_pars` as needed, some light documentation is provided in the code via comments.
  - In `define_datasets`, change the division angles, energies and centralities to process.
  - In `define_datasets`, the `entry_vals` list contains a sublist of parameter values defining the datasets to process. 
Use previous definitions (commented) as templates.
  - Run the script with `python calc_binom_slices.py`.
  - The output csv file will be saved in the `csv_path` directory, defined in `init_pars`.
- `analyze_binom_slice_plotter.py` runs all of the final analysis and plotting given the input csv of moments.
- To run, uncomment the appropriate function in `main`. In that function, update the input paths to point to the 
corresponding csv files produced by `calc_binom_slices.py`.
- This should produce all relevant plots.

## Running the Proton Clustering in Azimuthal Partitions Analysis on RCF

The code for the proton clustering in azimuthal partitions analysis was developed and run on a local machine.
However, most of these scripts can be equally well run on RCF, with the proper environment and logistics.
The following instructions outline the steps to run the analysis on RCF.

- Copy the `clone_and_source.sh` from the top directory of this repository to your working directory on RCF.
- In `clone_and_source.sh`, change the `INSTALL_DIR` variable to your working directory on RCF 
- (leave as $(pwd) to install things in current directory).
- Make sure `clone_and_source.sh` is executable: `chmod u+x clone_and_source.sh`
- Run `source clone_and_source.sh` to set up the environment and clone the repository. This will take a few minutes.
- Go to the `Python_Downstream` directory: `cd Proton_Fluctuations_Paper_Analysis/Python_Downstream`
- In the `init_pars` function of `calc_binom_slices.py`, change the `base_path` to point to the directory
  containing the `Data` and `Data_Mix` directories on RCF. Change the `csv_path` to the desired output directory on RCF.
- In the `define_datasets` function of `calc_binom_slices.py`, update the `all_energies` list to include the energies
to be run. In the `entry_vals` list, ensure the third argument of the sublist is set to the `set_group_name` from 
TreeReader.
- Run `calc_binom_slices.py` with `python calc_binom_slices.py`.
- The output csv file will be saved in the `csv_path` directory, defined in `init_pars`.
- Next, run `analyze_binom_slice_plotter.py` to perform the final analysis and plotting...

