import ast
import warnings

from os import listdir
from os.path import isfile, join, exists, basename, dirname

import torch
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from argparse import ArgumentParser

from src.utils.NewsArticle import NewsArticle
from src.utils.NewsEventMonitor import NewsEventMonitor
from src.models.PairBERT import PairBERT

warnings.simplefilter(action="ignore")

# ================================================
# Helper functions
# ================================================


def literal_converter(val):
    try:
        return ast.literal_eval(val)
    except Exception:
        return val


def create_dataframe(event_monitor):
    """Store all articles into the dataframe"""

    event_monitor.assign_events_to_articles()
    data = [
        article.to_array()[:-1]
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
        ],
    )
    df.index.name = "id"
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
        converters={"source": literal_converter},
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
    sim_th,
    time_th_in_days,
    time_metric,
    compare_th,
    compare_model_path,
    compare_ne,
    run_as_test=False,
    is_multilingual=False,
    use_gpu=False,
):
    if use_gpu and not torch.cuda.is_available():
        warnings.warn("GPU not available, using CPU")

    device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")

    compare_model = PairBERT.load_from_checkpoint(compare_model_path)

    # load the compare model onto the GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    compare_model = compare_model.to(device)

    event_monitor = NewsEventMonitor(
        sim_th=sim_th,
        time_th_in_days=time_th_in_days,
        time_metric=time_metric,
        compare_th=compare_th,
        compare_model=compare_model,
        compare_ne=compare_ne,
        is_multilingual=is_multilingual,
    )

    articles = load_articles(input_file, run_as_test)
    for article in tqdm(articles, desc=input_file.split("/")[-1]):
        # specify where we compare the articles
        event_monitor.update(article, device=device)

    df = create_dataframe(event_monitor)
    df.to_csv(output_file, encoding="utf-8", index=True)


# ================================================
# Main function
# ================================================


def main(args):
    if isfile(args.input_dir):
        input_dir = dirname(args.input_dir)
        files = [basename(args.input_dir)]
    else:
        input_dir = args.input_dir
        files = [f for f in listdir(input_dir) if isfile(join(input_dir, f))]

    # create the results directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    for file in tqdm(files, desc="Files"):
        if exists(f"{args.output_dir}/{file}") and not args.override:
            continue
        cluster_and_save_articles(
            input_file=f"{input_dir}/{file}",
            output_file=f"{args.output_dir}/{file}",
            sim_th=args.sim_th,
            time_th_in_days=args.time_th_in_days,
            time_metric=args.time_metric,
            compare_th=args.compare_th,
            compare_model_path=args.compare_model_path,
            compare_ne=args.compare_ne,
            is_multilingual=args.is_multilingual,
            run_as_test=args.test,
            use_gpu=args.use_gpu,
        )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--input_dir", default=None, type=str)
    parser.add_argument("--output_dir", default=None, type=str)
    parser.add_argument("--sim_th", default=0.8, type=float)
    parser.add_argument("--time_th_in_days", default=2, type=int)
    parser.add_argument("--time_metric", default="min", type=str)
    parser.add_argument("--compare_th", default=0.8, type=float)
    parser.add_argument(
        "--compare_model_path",
        default="./models/pairbert-multilingual-mpnet-base-v4.ckpt",
        type=str,
    )
    parser.add_argument("--compare_ne", default=True, type=bool)
    parser.add_argument("--is_multilingual", action="store_true")
    parser.add_argument("--use_gpu", action="store_true")
    parser.add_argument("--override", action="store_true")
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()

    main(args)
