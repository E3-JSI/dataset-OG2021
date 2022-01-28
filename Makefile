# specify the repository environment name
REPO_NAME := worldnews
CUDA_VERSION := 11.1

.ONESHELL:
SHELL=/bin/bash
CONDA_ACTIVATE=source $$(conda info --base)/etc/profile.d/conda.sh ; conda activate ; conda activate

# setup the experiment environment
setup: requirements.txt setup.py
	conda create --name $(REPO_NAME) python=3.8 pip --yes
	$(CONDA_ACTIVATE) $(REPO_NAME)
	pip install -e .



# setups the data collector
setup-data-collector: .env
	git submodule update --remote --merge
	$(CONDA_ACTIVATE) $(REPO_NAME)
	pip install -e ./services/data-collector
	cp ./.env ./services/data-collector


# collects the news articles
run-data-collection:
	cd ./scripts && bash -i ./run_news_collector.sh


# install pytorch in the experiment environment
pytorch: setup
	conda install -n $(REPO_NAME) pytorch torchvision torchaudio cpuonly -c pytorch --yes
	conda install -n $(REPO_NAME) pytorch-lightning torchmetrics=0.6.2 -c conda-forge --yes

# install pytorch in the experiment environment (with CUDA)
pytorch-cuda: setup
	conda install -n $(REPO_NAME) pytorch torchvision torchaudio cudatoolkit=$(CUDA_VERSION) -c pytorch -c nvidia --yes
	conda install -n $(REPO_NAME) pytorch-lightning torchmetrics=0.6.2 -c conda-forge --yes

# install jupyter extensions configurator
jupyter: setup
	conda install -n $(REPO_NAME) jupyter_nbextensions_configurator jupyterlab_execute_time -c conda-forge --yes

# clean the project
clean:
	rm -rf .eggs $(REPO_NAME).egg-info
	conda env remove -n $(REPO_NAME)
