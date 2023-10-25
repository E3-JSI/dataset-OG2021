# Scripts

This folder contains the scripts for:

1. setting up the project environment;

   ```bash
   bash setup_environment.sh {event-registry-api-key}
   ```

2. Collecting the raw news data from Event Registry;

   ```bash
   bash run_news_collector.sh
   ```


Here we explain how to recreate the experiment with our code.


1. Divide the dataset into concepts created by Event Registry using the following python script:

```
create_datasets.py
```
This script will create a new folder in data named concepts (data/concepts). When finished the folder will contain csv files which names represent the containing concept.

2. After that use:

``` 
PairBERT_labeling.py
```
This script need two parameters the input csv file (one of csv files created in the first step) and the output (where do you want the csv file to be saved).

3. Split dataset by concepts

```
02_split_by_concepts.py
```
Script partitions the entire dataset into distinct concepts (column concepts), creating a separate CSV file for each concept.

4. Now we can start with labeling

```
03_PairBERT_concepts_labeling.py
```
This script employs script named PairBERT_labeling to separately cluster articles within each concept into event-based clusters. Input of this script is the file (data/concepts) created by the script in the first step.

5. Merging labeled concepts in one file.

```
04_concepts_merging.py
```
This script merges together concepts into a single CSV file. Input of this script is the file where labečed concepts were saved by script ```03_PairBERT_concepts_labeling.py```.

