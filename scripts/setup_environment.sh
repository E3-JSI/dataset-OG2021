#!/bin/bash

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
# Prepare python environment
# ===============================================

# check if python is available
if [[ $(which python) == "" ]]; then
    echo "Python not installed on this machine"
    exit
fi;

# check if the repository environment is initialized
REPO_ENV="venv"
if [[ -d "../$REPO_ENV" ]]; then
    echo "Python environment '$REPO_ENV' already initialized"
else
    # create a new virtual environment with the installed python
    python -m venv ../venv
    # activate the environment
    . ../venv/bin/activate
fi;

# ===============================================
# Setup the news collector
# ===============================================

if [[ -d "../$REPO_ENV" ]]; then
    pip install -e ../news-collector
    echo "News collector initialized"
fi;

# ===============================================
# Prepare collector environment variables
# ===============================================

ER_API_KEY=$1
if [[ -z "$ER_API_KEY" ]]; then
    echo "Event Registry API Key not specified"
else
    echo "Copying the Event Registry API Key"
    # create the .env file with the API key as the content
    echo "API_KEY=$ER_API_KEY" > ../news-collector/.env
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
