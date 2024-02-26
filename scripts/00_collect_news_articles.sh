#!/bin/bash

# stop whole script when CTRL+C
trap "exit" INT

# ===============================================
# Check for conda
# ===============================================

# check if conda is available
if [[ $(which conda) == "" ]]; then
    echo "Conda not installed on this machine"
    exit
fi;

# ===============================================
# Prepare project environment
# ===============================================

# set repository environment
REPO_ENV="worldnews"
ENVS=$(conda env list | awk '{print $1}')

if [[ $ENVS = *"$REPO_ENV"* ]]; then
    # activate the environment
    conda activate "$REPO_ENV"
else
    echo "Project environment not setup. Please create the environment"
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

# TODO: prepare a list of query options to retrieve the data from
# QUERY FORMAT: date-start;date-end;comma-separated-concepts
declare -a QUERIES=(
    "2021-07-01;2021-08-20;olympic games,japan,basketball"
    "2021-07-01;2021-08-20;olympic games,japan,sport climbing"
    "2021-07-01;2021-08-20;olympic games,japan,swimming"
    "2021-07-01;2021-08-20;olympic games,japan,judo"
    "2021-07-01;2021-08-20;olympic games,japan,rowing"
    "2021-07-01;2021-08-20;olympic games,japan,skateboarding"
    "2021-07-01;2021-08-20;olympic games,japan,table tennis"
)

for QUERY in "${QUERIES[@]}"; do

    # turn e.g. "2021-05-01;2021-06-01;concepts" into
    # array ["2021-05-01", "2021-06-01", "concepts"]
    IFS=";" read -r -a ELEMENTS <<< "${QUERY}"

    # extract the values
    DATE_START="${ELEMENTS[0]}"
    DATE_END="${ELEMENTS[1]}"
    CONCEPTS="${ELEMENTS[2]}"

    if [[ $DATE_END == "TODAY" ]]; then
        DATE_END=`date +"%Y-%m-%d"`
    fi;


    for LANGUAGE in "${LANGUAGES[@]}"; do

        echo "START COLLECTING ARTICLES FOR THE LANGUAGE: $LANGUAGE"

        # define the start date and add the days
        CURR_DATE_START=$DATE_START

        # iteratively go through the time interval
        while [[ $CURR_DATE_START < $DATE_END ]]; do
            # calculate the current end date
            CURR_DATE_END=$(date +"%Y-%m-%d" -d "$CURR_DATE_START + 6 days")
            if [[ $CURR_DATE_END > $DATE_END ]]; then
                CURR_DATE_END=$DATE_END
            fi;
            # construct the articles path
            DATE_START_PATH="${CURR_DATE_START//-/}"
            DATE_END_PATH="${CURR_DATE_END//-/}"
            TMP_CONCEPTS="${CONCEPTS// /_}"
            ARTICLES_PATH="../data/raw/${LANGUAGE}/${TMP_CONCEPTS//,/&}-${DATE_START_PATH}-${DATE_END_PATH}.jsonl"

            if [[ ! -f "$ARTICLES_PATH" ]]; then
                echo "ARTICLES NOT YET COLLECTED"
                # get the articles fitting the query parameters
                collect articles \
                    --max_repeat_request=5 \
                    --concepts="$CONCEPTS" \
                    --languages=$LANGUAGE \
                    --date_start=$CURR_DATE_START \
                    --date_end=$CURR_DATE_END \
                    --save_to_file=$ARTICLES_PATH \
                    --verbose=True
            fi;

            if [[ -f "$ARTICLES_PATH" ]]; then
                # output the number of articles in the generated file
                echo "COLLECTED ARTICLES IN FILE: $ARTICLES_PATH"
                awk '{print NF}' "$ARTICLES_PATH" | sort -nu | tail -n 1
            fi;

            # IMPORTANT: update the current date to the next week
            CURR_DATE_START=$(date +"%Y-%m-%d" -d "$CURR_DATE_START + 7 days")
        done;

        echo "ENDED COLLECTING ARTICLES FOR THE LANGUAGE: $LANGUAGE"

    done;
done;
