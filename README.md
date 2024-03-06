# OG2021: The 2021 Olympic Games data set

[![DOI](https://zenodo.org/badge/416290920.svg)](https://zenodo.org/doi/10.5281/zenodo.10785670)


This repository contains the source code for creating the 2021 Tokyo Olympic Games
data set (OG2021), a multilingual corpus of annotated news articles used for
evaluating clustering algorithms. The data set is a collection of 10.940 articles
in nine languages reporting the 2021 Tokyo Olympics events. The articles are grouped
into 1.350 clusters.

## 📚 Data

The data set is available on [clarin.si][clarin-si]. Specifically, there are two versions:

[Public data set][data-public]. Due to legal restrictions, the public data set does not contain the body of the articles. Consider using the research data set, if you want to include the article body. The data set is behind the CC BY-NC-ND 4.0 license.

[Research data set][data-research]. The research data set contains all of the article attributes, but is restricted only for research purposes. The data set is behind the Research license


### Data Format

The data is in the csv format. Each line contains one article. The columns are:

- **id**: The ID of the news article.
- **title**: The title of the article.
- **body**: The body of the article (available only in the research data set version)
- **lang**: The language in which the article is written. Can be one of nine values.
- **source**: The news publisher's name.
- **published_at**: The date and time when the article was published. The dates range between 2021-07-01 and 2021-08-14.
- **URL**: The URL location of the news article.
- **cluster_id**: The ID of the cluster the article is a member of.

**Language(s):** English, Portuguese, Spanish, French, Russian, German, Slovenian, Arabic, Chinese

## 🔎 Reference

If the data set was used for your research, please provide the following reference:

When using the research data set, use the following reference:

```bibtex
 @misc{11356/1921,
 title = {The news articles reporting on the 2021 Tokyo Olympics data set {OG2021} (research)},
 author = {Novak, Erik and Calcina, Erik and Mladeni{\'c}, Dunja and Grobelnik, Marko},
 url = {http://hdl.handle.net/11356/1921},
 note = {Slovenian language resource repository {CLARIN}.{SI}},
 copyright = {{CLARIN}.{SI} Licence {ACA} {ID}-{BY}-{NC}-{INF}-{NORED} 1.0},
 issn = {2820-4042},
 year = {2024} }
```


When using the public data set, use the following reference:
```bibtex
 @misc{11356/1922,
 title = {The news articles reporting on the 2021 Tokyo Olympics data set {OG2021} (public)},
 author = {Novak, Erik and Calcina, Erik and Mladeni{\'c}, Dunja and Grobelnik, Marko},
 url = {http://hdl.handle.net/11356/1922},
 note = {Slovenian language resource repository {CLARIN}.{SI}},
 copyright = {Creative Commons - Attribution-{NonCommercial}-{NoDerivatives} 4.0 International ({CC} {BY}-{NC}-{ND} 4.0)},
 issn = {2820-4042},
 year = {2024} }
```


## 📣 Acknowledgments

This work is developed by [Department of Artificial Intelligence][ailab] at [Jozef Stefan Institute][ijs].

This work is supported by the Slovenian Research Agency and the H2020 project
[Humane AI Network][project] (grant no. 952026).


<details>
  <summary>📝 Click here to see the technical details</summary>

## ☑️ Requirements

Before starting the project make sure these requirements are available:

- [python]. For setting up your research environment and python dependencies (Python 3.8 or higher).
- [git]. For versioning your code.

## 🛠️ Setup

### Create a python environment

First create the virtual environment where all the modules will be stored.
Using the `venv` command, run the following commands:

```bash
# create a new virtual environment
python -m venv venv

# activate the environment (UNIX)
source ./venv/bin/activate

# activate the environment (WINDOWS)
./venv/Scripts/activate

# deactivate the environment (UNIX & WINDOWS)
deactivate
```

### Install

To install the requirements, run:

```bash
pip install -e .
```

## 🗃️ Data Retrieval


### 🔍️ Collect the data via Event Registry API

To collect the data via the [Event Registry API], follow the next steps:

1. **Login into the Event Registry.** Create a user account in the Event Registry
   service and retrieve the API key that has assigned to it. The API key can be
   found in `Settings > Your API key`.

2. **Create the Environment File.** Create a copy of the `.env.example` file
   named `.env` and replace the `API_KEY` value with the API key assigned to
   your user account.

3. **Install the Data Collector.** Install the data collector using the
   following commands:

   ```bash
   # activate the environment
   source ./venv/bin/activate

   # pull the git submodules
   git submodule update --remote --merge

   # install the data collector module
   pip install -e ./services/data-collector

   # copy the environment file
   cp ./.env ./services/data-collector
   ```

4. **Collect the News Articles.** To collect the news, run the following commands:

   ```bash
   # move into the scripts folder
   cd ./scripts
   # start the news article collection
   bash -i collect_news_articles.sh
   ```

The data should be collected and stored in the `/data` folder.



## 🚀 Running scripts

To run the scripts follow the next steps:

### Data cleanup

To prepare and cleanup the data, run the following script:

```bash
python scripts/01_data_cleanup.py \
   --raw_dir ./data/raw \
   --results ./data/processed/articles.jsonl
```
This will retrieve the raw files found in the `raw_dir` folder, clean them up and store them in the `results` file.


### Split data into groups

The processed `articles.jsonl` contains all of the articles together. However, each article is associated with a set of concepts used to retrieve them from Event Registry (during the news article collection step). To ensure the data clustering is as efficient as possible, we need to split the articles into groups. This is done with the following script:

```bash
python scripts/02_data_concepts_split.py \
   --articles_dir ./data/processed \
   --concepts_dir ./data/processed/concepts
```

### Monolingual news article clustering

To perform monolingual clustering of the articles, run the following script:

```bash
python scripts/03_article_clustering.py \
   --concepts_dir ./data/processed/concepts \
   --events_dir ./data/processed/mono
```

### Multilingual news event clustering

To perform multilingual clustering, i.e. group clusters created in the previous step, run the following script:

```bash
python scripts/04_cluster_merging.py \
   --mono_events_dir ./data/processed/mono
   --multi_events_dir ./data/processed/multi
```


### Manual news event cleanup and evaluation

Each concept data set is manually evaluated. We defined the manual evaluation procedure in the notebook [01-individual-manual-evaluation.ipynb](notebooks/01-individual-manual-evaluation.ipynb). There, we store the evaluation results in the `manual_eval` folder.

Afterwards, we join the clusters and store the result in the `manual_join` folder.

```bash
python scripts/05_data_merge.py \
   --manual_eval_dir ./data/processed/manual_eval \
   --merge_file_path ./data/processed/manual_join/og2021.csv
```

Since individual concept data sets might contain clusters reporting across multiple data sets, we need to manually merge them. This is done in the notebook [02-group-manual-evaluation.ipynb](notebooks/02-group-manual-evaluation.ipynb), resulting in the final data set stored in the `data/final` folder.


### Data set statistics
The data set statistics and visualizations are computed in the notebook [03-final-dataset-analysis.ipynb](notebooks/03-final-dataset-analysis.ipynb).


</details>





[python]: https://www.python.org/
[git]: https://git-scm.com/

[Event Registry API]: https://eventregistry.org/

[clarin-si]: https://www.clarin.si/
[data-public]: http://hdl.handle.net/11356/1921
[data-research]: http://hdl.handle.net/11356/1921

[ailab]: http://ailab.ijs.si/
[ijs]: https://www.ijs.si/
[project]: https://www.humane-ai.eu/


[cc-by]: http://creativecommons.org/licenses/by/4.0/
[cc-by-shield]: https://img.shields.io/badge/License-CC%20BY-lightgrey.svg
