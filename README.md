# WorldNews Data Set

This repository contains the source code for creating the WorldNews Data Set,
the data set containing news articles. The articles are in different languages
and cover various topics.

The goal of this project is to create a multilingual news article corpus used
for news stream clustering and topic classification.

# Initialization

To prepare the project run the following commands:

```bash
# move into the scripts folder
cd ./scripts

# setup the project environment
# NOTE: the event registry API key is required
#       only for collecting news articles
bash -i setup_environment.sh {event-registry-api-key}
```

The above command will create a new conda environment called `worldnews`,
and install all dependencies written in `requirements.txt`.

## Install Pytorch and Pytorch-Lightning

```bash
# activate the project environment
conda activate worldnews

# install pytorch
conda install pytorch cudatoolkit=11.3 -c pytorch
conda activate worldnews

# install pytorch lightning
conda install pytorch-lightning -c conda-forge
```

## Get Data

### Get collected data via DVC

In this project we use [DVC][dvc] for tracking the changes done to the data sets.
To get the data for the project do the following:

1. Ask the project admin to generate a new user on the machine where the data
   is stored.

2. Next, run the following two commands:

   ```bash
   # locally store the username provided by the admin
   dvc remote modify --local ssh-storage user {username}

   # locally store the password provided by the admin
   dvc remote modify --local ssh-storage password {password}
   ```

3. Finally, run the command:

   ```bash
   # pulls the data from the remote location
   dvc pull
   ```

This will create a new folder `/data` which will contain all of the data
for the project.

### Collect the data via Event Registry API

To collect the data via the Event Registry API, follow the next steps:

- **Initialize data-collector.** This step should be performed during the
  initialization step.
- **Run the data-collector.** Execute the following commands in the terminal
  ```bash
  # move into the scripts folder
  cd ./scripts
  # start the news collector
  bash -i run_news_collector.sh
  ```

The data should be collected and stored in the `/data` folder.

[dvc]: https://dvc.org/
