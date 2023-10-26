import pandas as pd
from tqdm import tqdm
from math import floor
import ast


from argparse import ArgumentParser

from src.utils.NewsArticle import NewsArticle
from src.utils.NewsEventMonitorPairBERT import NewsEventMonitor
from src.utils.PairBERT2 import PairBERT

import warnings

"""
This script employs PairBERT to cluster articles into event-based clusters.
"""


def main(hparams):
    
    warnings.simplefilter(action='ignore')
    
    df_input = pd.read_csv(
        hparams.input,
        dtype={
            "id": "int",
            "dateTime": "str",
            "title": "str",
            "body": "str",
            "cluster": "int",
        },
        converters={'source': ast.literal_eval},
        parse_dates=["dateTime"],
        index_col = False,
    )
    
    #Some news dont have titles (end with an error) we drop those news
    df_input = df_input.drop(df_input[df_input['title'].isnull() == True].index)
    
    df_dict = df_input.to_dict('records')
    
    articles = [NewsArticle(article) for article in df_dict]
    model = PairBERT.load_from_checkpoint("../models/pairbert-multi-v3.ckpt")
    
    sim_threshold = 0.5
    time_threshold_in_days = 3
    time_compare_stat = "min"

    event_monitor = NewsEventMonitor(
        sim_threshold=sim_threshold,
        time_threshold_in_days=time_threshold_in_days,
        time_compare_stat=time_compare_stat,
        model=model    
    )
    
    for article in tqdm(articles, desc="Article Feed"):
        event_monitor.update(article)
    
    events = event_monitor.get_events()
    
    print("Creating dataset...")
    df_clustered = event_monitor.get_dataframe()
    
    df_clustered.to_csv(hparams.output, encoding='utf-8', index=True)
    


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--input", default=None) #--> path of the non alebeld news file
    parser.add_argument("--output", default=None) #--> path for storing labeled file
    args = parser.parse_args()

    main(args)