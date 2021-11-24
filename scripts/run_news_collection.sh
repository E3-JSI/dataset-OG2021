#!/bin/bash

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
if [[ ! -d "../data/news" ]]; then
    mkdir ../data/news
fi;

# ===============================================
# Navigate into the news collector submodule
# ===============================================

# navigate into the news collector
cd ../news-collector

# ===============================================
# Collect data
# ===============================================

# TODO: prepare a list of query options to retrieve the data from
DATE_START="2020-01-01"
LANGUAGES="eng,zho,spa,ara,por,fra,jpn,rus,deu,slv"

CONCEPTS=(
    "Luka Dončić"
)

for (( i = 0; i < ${#CONCEPTS[@]}; ++i )); do
    # prepare the files and folders
    EVENT_FILE="../data/news/events/${CONCEPTS[i]// /_}.jsonl"
    ARTICLES_FOLDER="../data/news/articles/${CONCEPTS[i]// /_}"

    # TODO: figure out if we want to retrieve the articles or do
    # TODO: we want to retrieve the events and the their articles

    # get the events mentioning the concepts
    collect events \
        --max_repeat_request=5 \
        --concepts="${CONCEPTS[i]}" \
        --languages="$LANGUAGES" \
        --date_start="$DATE_START" \
        --save_to_file="$EVENT_FILE"


    if [[ -d "$EVENT_FILE" ]]; then
        # TODO: get the articles of the events acquired with the above command
        collect event_articles_from_file \
            --max_repeat_request=5 \
            --event_ids_file="$EVENT_FILE" \
            --save_to_file="$ARTICLES_FOLDER"
    else
        echo "Event file non-existant! Parameters:
                --concepts='${CONCEPTS[i]}'
                --languages='$LANGUAGES'
                --date_start='$DATE_START'
        "
    fi;

done
