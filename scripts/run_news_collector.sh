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
    "ever given,container ship,suez canal"
)

for i in ${!CONCEPTS[@]}; do
    # get the current concept
    CONCEPT=${CONCEPTS[$i]}

    for j in ${!LANGUAGES[@]}; do
        # get the current language
        LANGUAGE=${LANGUAGES[$j]}

        # # prepare the files and folders
        # EVENTS_PATH="../data/news/${LANGUAGE}/events/${CONCEPT// /_}.jsonl"
        # ARTICLES_PATH="../data/news/${LANGUAGE}/articles/${CONCEPT// /_}"

        # TODO: figure out if we want to retrieve the articles or do
        # TODO: we want to retrieve the events and the their articles
        ARTICLES_PATH="../data/news/${LANGUAGE}/${CONCEPT// /_}.jsonl"

        # get the articles fitting the query parameters
        collect articles \
            --max_repeat_request=5 \
            --concepts="$CONCEPT" \
            --languages=$LANGUAGE \
            --date_start=$DATE_START \
            --save_to_file=$ARTICLES_PATH


        # # get the events mentioning the concepts
        # collect events \
        #     --max_repeat_request=5 \
        #     --concepts="$CONCEPT" \
        #     --languages=$LANGUAGE \
        #     --date_start=$DATE_START \
        #     --save_to_file=$EVENTS_PATH


        # if [[ -f $EVENTS_PATH ]]; then
        #     # get the articles of the events acquired with the above command
        #     collect event_articles_from_file \
        #         --max_repeat_request=5 \
        #         --event_ids_file=$EVENTS_PATH \
        #         --save_to_file=$ARTICLES_PATH
        # else
        #     echo "Event file non-existant! Parameters:
        #             --concepts='$CONCEPT'
        #             --languages='$LANGUAGE'
        #             --date_start='$DATE_START'
        #     "
        # fi;
    done
done
