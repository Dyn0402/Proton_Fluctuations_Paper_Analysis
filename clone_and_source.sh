#!/bin/bash

# Exit on any error
#set -e

rm -rf Proton_Fluctuations_Paper_Analysis
rm -rf partition_env

# Set the installation directory as a variable
INSTALL_DIR=$(pwd)

# Activate dylan's conda environment
source /star/u/dneff/Software/anaconda3/bin/python

# Clone the repositories
git clone https://github.com/Dyn0402/Proton_Fluctuations_Paper_Analysis.git

# Create the virtual environment in the current directory
python -m venv --without-pip --clear "$INSTALL_DIR/partition_env"
python -m venv "$INSTALL_DIR/partition_env"

# Activate the virtual environment
source "$INSTALL_DIR/partition_env/bin/activate"

# Export the PYTHONPATH
export PYTHONPATH="$INSTALL_DIR/partition_env/lib/python3.9/site-packages"

# Install requirements
pip install --upgrade pip
pip install -r $INSTALL_DIR/Proton_Fluctuations_Paper_Analysis/Python_Downstream/requirements.txt
