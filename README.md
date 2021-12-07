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

## Get Data with DVC

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

[dvc]: https://dvc.org/
