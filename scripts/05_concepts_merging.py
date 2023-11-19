import os
from os import listdir
from os.path import isfile, join
import pandas as pd


"""
This script merges together concepts into a single CSV file.
"""

mypath = "data/labeled2"
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]

df = pd.DataFrame(columns=['title','body', 'concept','time','lang','eventUri','url','source','cluster_id'])

for file in onlyfiles:
    currentDF = pd.read_csv(f"data/labeled2/{file}")
    print("Currently merging:", file)
    if df.empty:
        df = currentDF
    else:
        currentDF["cluster_id"] = currentDF['cluster_id'].apply(lambda x: x + (df.cluster_id.max() + 1))
        frames = [df, currentDF]
        df = pd.concat(frames)

df.reset_index(drop=True, inplace=True)

df = df.rename(columns={'Unnamed: 0': 'Concept Index'})

df.to_csv("data/labeled2/entire_dataset.csv", encoding='utf-8', index=True)