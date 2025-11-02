#!/bin/bash

# Exit on any error
#set -e

rm -rf Proton_Fluctuations_Paper_Analysis
rm -rf partition_env

# Set the installation directory as a variable
#INSTALL_DIR=$(pwd)

# Activate dylan's conda environment
source /star/u/dneff/Software/anaconda3/bin/activate

# Clone the repositories
git clone https://github.com/Dyn0402/Proton_Fluctuations_Paper_Analysis.git

# Put dylan's python site-packages ahead of star's
export PYTHONPATH="/star/u/dneff/Software/anaconda3/lib/python3.9/site-packages"
