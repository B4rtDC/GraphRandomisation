#!/bin/bash

# 0. conda environment name (defaults to "graphrandomisation", change if required).
CONDA_ENV_NAME="graphrandomisation"

# 1. Create the conda environment (or use the commented command for a custom name)
conda env create -f ./graphrandomisation.yml -n $CONDA_ENV_NAME

# 2. Fetch the path to the virtual environment
# This uses the `conda info` command and parses the output for the correct path
# The --json flag makes this easier and more reliable across systems
CONDA_PREFIX=$(conda env list --json | jq -r --arg env_name "$CONDA_ENV_NAME" '.envs[] | select(contains($env_name))' | tr -d '\n')

# Check if CONDA_PREFIX is empty
if [ -z "$CONDA_PREFIX" ]; then
    echo "Error: Unable to find the conda environment path."
    exit 1
fi

# 3. Activate the conda environment
conda activate $CONDA_ENV_NAME

# 4. Start Julia, set environment variable, and install/build PyCall accordingly
julia --project=. -e '
    using Pkg
    ENV["PYTHON"] = "'"$CONDA_PREFIX/bin/python"'" # Properly escape the environment variable
    @info """using this path: $(ENV["PYTHON"])"""
    Pkg.add("PyCall")
    Pkg.build("PyCall")
    using PyCall
    plaw = pyimport("powerlaw")
    @info "PyCall setup done, all OK"
'

# 5. Add additional dependencies 
julia --project=. -e '
    using Pkg
    Pkg.add([   "BenchmarkTools";
                "ProgressMeter"
                "CSV";
                "DataFrames";
                "JLD2";
                "JSON";
                "StatsBase";
                "Combinatorics";
                "HypothesisTests";
                "Graphs";
                "MaxEntropyGraphs";
                "Plots";
                "StatsPlots";
                "Measures";
                "LaTeXStrings";
                "Distributed"
                ])'


