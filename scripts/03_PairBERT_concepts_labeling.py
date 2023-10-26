from os import listdir
from os.path import isfile, join
import os
import pandas as pd
import time

"""
This script employs script named PairBERT_labeling to separately cluster articles within each concept into event-based clusters.
"""

#From path (mypath) get all non labeled csv files
mypath = "data/concepts"
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]

#Label all colected files
for file in onlyfiles:
    print("Currently labeling",file)
    os.system('python3 ' + f"scripts/PairBERT_labeling.py --input data/concepts/{file} --output data/labeled2/{file}")
    time.sleep(20)