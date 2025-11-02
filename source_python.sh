#!/bin/bash

# Exit on any error
#set -e

# Activate dylan's conda environment
source /star/u/dneff/Software/anaconda3/bin/activate

# Put dylan's python site-packages ahead of star's
export PYTHONPATH="/star/u/dneff/Software/anaconda3/lib/python3.9/site-packages"