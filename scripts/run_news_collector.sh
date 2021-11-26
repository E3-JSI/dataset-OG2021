#!/bin/bash

# stop whole script when CTRL+C
trap "exit" INT

# ===============================================
# Check for python
# ===============================================

# check if conda is available
if [[ $(which python) == "" ]]; then
    echo "Python not installed on this machine"
    exit
fi;

# ===============================================
# Prepare python environment
# ===============================================

# set repository environment
REPO_ENV="venv"
if [[ -d "../$REPO_ENV" ]]; then
    # activate the environment
    . ../venv/bin/activate
else
    echo "Python environment not setup. Please run 'setup_environment.sh' script"
    exit
fi;

# ===============================================
# Prepare the data folder
# ===============================================

# create a new folder
if [[ ! -d "../data/raw" ]]; then
    mkdir ../data/raw
fi;

# ===============================================
# Collect data
# ===============================================

# TODO: prepare a list of query options to retrieve the data from
DATE_START="2020-01-01"
declare -a LANGUAGES=(
    "eng"
    "zho"
    "spa"
    "ara"
    "por"
    "fra"
    "jpn"
    "rus"
    "deu"
    "slv"
)

declare -a CONCEPTS=(
    #"Kobe Bryant,Helicopter,Basketball"
    #"Container ship,Suez Canal"
    "Pandora Papers"
    "2020-2021 global chip shortage"
    "Hong Kong,Demonstration (Protest)"
    "Basketball,National Basketball Association (NBA)"
    "EuroBasket"
    "UEFA Champions League,Association football,Match"
)

for i in ${!CONCEPTS[@]}; do
    # get the current concept
    CONCEPT=${CONCEPTS[$i]}

    for j in ${!LANGUAGES[@]}; do
        # get the current language
        LANGUAGE=${LANGUAGES[$j]}

        ARTICLES_PATH="../data/raw/${LANGUAGE}/${CONCEPT// /_}.jsonl"

        # get the articles fitting the query parameters
        collect articles \
            --max_repeat_request=5 \
            --concepts="$CONCEPT" \
            --languages=$LANGUAGE \
            --date_start=$DATE_START \
            --save_to_file=$ARTICLES_PATH

        awk '{print NF}' "$ARTICLES_PATH" | sort -nu | tail -n 1

    done
done
