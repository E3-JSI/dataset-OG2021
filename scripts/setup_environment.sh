#!/bin/bash

# stop whole script when CTRL+C
trap "exit" INT

# ===============================================
# Retrieve all gitmodules
# ===============================================

# check if python is available
if [[ $(which git) == "" ]]; then
    echo "Git not installed on this machine"
    exit
fi;

git submodule update --remote --merge

# ===============================================
# Prepare conda environment
# ===============================================

# check if conda is available
if [[ $(which conda) == "" ]]; then
    echo "Conda not installed on this machine"
    exit
fi;

# # check if the repository environment is initialized
# REPO_ENV="venv"
# if [[ -d "../$REPO_ENV" ]]; then
#     echo "Python environment '$REPO_ENV' already initialized"
# else
#     # create a new virtual environment with the installed python
#     python -m venv ../venv
#     # activate the environment
#     . ../venv/bin/activate
# fi;

# check if the repository environment is initialized
REPO_ENV="worldnews"
ENVS=$(conda env list | awk '{print $1}')

if [[ $ENVS = *"$REPO_ENV"* ]]; then
    echo "Python environment '$REPO_ENV' already initialized"
else
    # create a new virtual environment with the installed python
    conda create --name "$REPO_ENV" python=3.8 pip --yes
fi;

conda activate $REPO_ENV

# ===============================================
# Setup the project repository
# ===============================================

pip install -e ..
echo "Project repository initialized"

# ===============================================
# Setup the news collector
# ===============================================

pip install -e ../services/data-collector
echo "News collector initialized"

# ===============================================
# Prepare collector environment variables
# ===============================================

ER_API_KEY=$1
if [[ -z "$ER_API_KEY" ]]; then
    echo "Event Registry API Key not specified"
else
    echo "Copying the Event Registry API Key"
    # create the .env file with the API key as the content
    echo "API_KEY=$ER_API_KEY" > ../services/data-collector/.env
fi;

# ===============================================
# Create the data folder
# ===============================================

# create the data folder
if [[ -d "../data" ]]; then
    echo "Data folder already exists. Skipping this step..."
else
    mkdir ../data
    echo "Data folder created"
fi;
