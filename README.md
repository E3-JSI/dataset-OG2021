# Worldnews

This repository contains the source code for creating the worldnews a data set
containing annotated news articles for evaluating clustering algorithms. The
articles are in different languages and cover various topics.

The goal of this project is to create a multilingual news article corpus used
for news stream clustering and topic classification.

## ☑️ Requirements

Before starting the project make sure these requirements are available:

- [conda][conda]. For setting up your research environment and python dependencies.
- [dvc][dvc]. For versioning your data.
- [git][git]. For versioning your code.

## 🛠️ Setup

### Create a python environment

First create the virtual environment where all the modules will be stored.

#### Using virtualenv

Using the `virtualenv` command, run the following commands:

```bash
# install the virtual env command
pip install virtualenv

# create a new virtual environment
virtualenv -p python ./.venv

# activate the environment (UNIX)
./.venv/bin/activate

# activate the environment (WINDOWS)
./.venv/Scripts/activate

# deactivate the environment (UNIX & WINDOWS)
deactivate
```

#### Using conda

Install [conda][conda], a program for creating python virtual environments. Then run the following commands:

```bash
# create a new virtual environment
conda create --name worldnews python=3.8 pip

# activate the environment
conda activate worldnews

# deactivate the environment
deactivate
```

### Install

To install the requirements run:

```bash
pip install -e .
```

## 🗃️ Data

### 🦉 Using DVC

We use [DVC][dvc] for tracking the changes done to the data sets.
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

### 🔍️ Collect the data via Event Registry API (required conda environment)

To collect the data via the [Event Registry API][er], follow the next steps:

1. **Login into the Event Registry.** Create a user account in the Event Registry
   service and retrieve the API key that has assigned to it. The API key can be
   found in `Settings > Your API key`.

2. **Create the Environment File.** Create a copy of the `.env.example` file
   named `.env` and replace the `API_KEY` value with the API key assigned to
   your user account.

3. **Install the Data Collector.** Install the data collector using the
   following commands:

   ```bash
   # activate the environment (example shows the conda approach)
   conda activate worldnews

   # pull the git submodules
   git submodule update --remote --merge

   # install the data collector module
   pip install -e ./services/data-collector

   # copy the environment file
   cp ./.env ./service/data-collector
   ```

4. **Collect the News Articles.** To collect the news, run the following commands:

   ```bash
   # move into the scripts folder
   cd ./scripts
   # start the news article collection
   bash -i collect_news_articles.sh
   ```

The data should be collected and stored in the `/data` folder.

## 📣 Acknowledgments

This work is developed by [Department of Artificial Intelligence][ailab] at [Jozef Stefan Institute][ijs].

This work is supported by the Slovenian Research Agency and the H2020 project
Humane AI Network (grant no. 952026).


[python]: https://www.python.org/
[conda]: https://www.anaconda.com/
[git]: https://git-scm.com/
[dvc]: https://dvc.org/
[er]: https://eventregistry.org/

[ailab]: http://ailab.ijs.si/
[ijs]: https://www.ijs.si/