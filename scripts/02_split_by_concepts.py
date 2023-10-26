import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

"""
This script partitions the entire dataset into distinct concepts (column concepts), creating a separate CSV file for each concept.
"""

# import the dataset loader
from src.data.dataset import load_dataset, DATA_PATHS

# load the raw articles
dataset = load_dataset(DATA_PATHS["processed"])

# Create dataframe
data = pd.DataFrame(dataset)

# List of all unique concepts
unique_concept_list = data['concepts'].drop_duplicates().tolist()

for x in range(len(unique_concept_list)):
    print("Next concept is:",unique_concept_list[x])
    data = []
    data = pd.DataFrame(dataset)

    selection = unique_concept_list[x]  #selecting only where concepts keys in x
    mask = data.concepts.apply(lambda x: all(item in x for item in selection))
    data = data[mask]
    
    name = "_".join(str(y) for y in unique_concept_list[x])
    file = "data/concepts/"
    save_to = file + name + ".csv"
    
    print("saving to....:", save_to)
    data.to_csv(save_to, encoding='utf-8', index=True)