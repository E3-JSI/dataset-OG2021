import ast
import warnings

from os import listdir
from os.path import isfile, join, exists

import torch
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from argparse import ArgumentParser

from src.utils.NewsArticle import NewsArticle
from src.utils.NewsEventMonitor import NewsEventMonitor
from src.models.PairBERT import PairBERT

warnings.simplefilter(action="ignore")


def create_dataframe(event_monitor):
    """Store all articles into the dataframe"""

    event_monitor.assign_events_to_articles()
    data = [
        article.to_array()
        for event in tqdm(event_monitor.events, desc="Event prep")
        for article in event.articles
    ]
    df = pd.DataFrame(
        data,
        columns=[
            "title",
            "body",
            "lang",
            "source",
            "dateTime",
            "url",
            "uri",
            "eventUri",
            "concepts",
            "clusterId",
            "namedEntities",
            "wikiConcepts",
        ],
    )
    return df


def load_articles(input_file, run_as_test):
    df = pd.read_csv(
        input_file,
        dtype={
            "id": "int",
            "title": "str",
            "body": "str",
            "lang": "str",
            "dateTime": "str",
            "uri": "str",
            "url": "str",
            "concepts": "str",
        },
        converters={"source": ast.literal_eval},
        parse_dates=["dateTime"],
        index_col=False,
    )

    # Some news dont have titles (end with an error) we drop those news
    df = df.drop(df[df["title"].isnull()].index)
    df = df[:100] if run_as_test else df

    # create the news article list
    articles = [
        NewsArticle(article)
        for article in tqdm(df.to_dict("records"), desc="File load")
    ]
    return articles


def cluster_and_save_articles(
    input_file,
    output_file,
    sim_threshold,
    time_threshold_in_days,
    time_compare_stat,
    compare_threshold,
    compare_model_path,
    compare_ne,
    run_as_test=False,
):
    compare_model = PairBERT.load_from_checkpoint(compare_model_path)

    # load the compare model onto the GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    compare_model = compare_model.to(device)

    event_monitor = NewsEventMonitor(
        sim_threshold=sim_threshold,
        time_threshold_in_days=time_threshold_in_days,
        time_compare_stat=time_compare_stat,
        compare_threshold=compare_threshold,
        compare_model=compare_model,
        compare_ne=compare_ne,
    )

    articles = load_articles(input_file, run_as_test)
    for article in tqdm(articles, desc=input_file.split("/")[-1]):
        # specify where we compare the articles
        event_monitor.update(article, device=device)

    df = create_dataframe(event_monitor)
    df.to_csv(output_file, encoding="utf-8", index=True)


def main(args):
    concept_files = [
        f for f in listdir(args.concepts_dir) if isfile(join(args.concepts_dir, f))
    ]

    # create the results directory
    Path(args.mono_events_dir).mkdir(parents=True, exist_ok=True)
    for file in tqdm(concept_files, desc="Files"):
        if exists(f"{args.mono_events_dir}/{file}"):
            continue
        cluster_and_save_articles(
            input_file=f"{args.concepts_dir}/{file}",
            output_file=f"{args.mono_events_dir}/{file}",
            run_as_test=args.test,
            sim_threshold=args.sim_threshold,
            time_threshold_in_days=args.time_threshold_in_days,
            time_compare_stat=args.time_compare_stat,
            compare_threshold=args.compare_threshold,
            compare_model_path=args.compare_model_path,
            compare_ne=args.compare_ne,
        )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--concepts_dir", default=None, type=str)
    parser.add_argument("--mono_events_dir", default=None, type=str)
    parser.add_argument("--sim_threshold", default=0.8, type=float)
    parser.add_argument("--time_threshold_in_days", default=1, type=int)
    parser.add_argument("--time_compare_stat", default="min", type=str)
    parser.add_argument("--compare_threshold", default=0.8, type=float)
    parser.add_argument(
        "--compare_model_path",
        default="./models/pairbert-multilingual-mpnet-base-v2.ckpt",
        type=str,
    )
    parser.add_argument("--compare_ne", default=True, type=bool)
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()

    main(args)
