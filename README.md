# WorldNews Data Set

This repository contains the source code for creating the WorldNews Data Set,
the data set containing news articles. The articles are in different languages
and cover various topics.

The goal of this project is to create a multilingual news article corpus used
for news stream clustering and topic classification.

# Initialization

To prepare the project run the following commands:

```bash

cd ./scripts
# setup the project environment
# NOTE: the event registry API key is required
#       only for collecting news articles
bash setup_environment.sh {event-registry-api-key}
```

## Install Pytorch and Pytorch-Lightning

```bash
# activate the project environment
. ./venv/bin/activate
# install pytorch
pip install torch==1.10.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
# install pytorch lightning
pip install pytorch-lightning
```

## Get Data

To get the data, first ask the project admin to generate a new
user on the machine where the data is stored.

Next, run the following two commands:

```bash
# locally store the username provided by the admin
dvc remote modify --local ssh-storage user {username}
# locally store the password provided by the admin
dvc remote modify --local ssh-storage password {password}
```

Finally, run the command:

```bash
# pulls the data from the remote location
dvc pull
```

# Description

## News Stream Clustering

The original idea of this data set is to provide a gold standard data set for
benchmarking news stream clustering algorithms. To each article an event ID is
assigned; articles with the same event ID are describing the same event.

### Cluster labelling process

The labelling process is performed using data from Event Registry. The system
generates event clusters using machine learning. However, these clusters are not
of the best quality since it does not find all of the articles that are about the
same event due to the linguistic differences and machine learning errors. Hence,
we would refine the clusters using two tasks which can be performed in parallel.

1. **Event Cluster Cleanup.** In this task, the event clusters would be cleaned
   by identifying the articles that have a higher probability of not matching
   with the rest of the articles in the cluster (we name them **Possible Rogue Articles**).
   We then take the event representative article and the Possible Rogue Articles
   and ask the user to confirm if the article is rogue or not. Using the labels
   we would then run a script that would remove the Rogue Articles from the events.

2. **Event Cluster Merge.** This task focuses on merging multiple event clusters
   into a single one. We would first identify potential cluster merges (which
   were not merged due to various differences). The user is then given the
   representative articles of both event clusters with the task of identifying
   if the articles are about the same event. If yes, the events are merged.
   Otherwise, no changes are done.

## Topic Classification

The article clusters were additionally labeled with topics:

> TODO: Update topics

- POLITICS
- SPORTS
- CULTURE
- BUSINESS
- TECH
- LIFESTYLE
- ENTERTAINMENT

### Topic labelling process

> TODO: Update topics
