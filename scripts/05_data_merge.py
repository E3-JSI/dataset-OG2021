import ast
import pandas as pd

from os import listdir
from os.path import isfile, join

from tqdm import tqdm
from pathlib import Path
from argparse import ArgumentParser

from src.utils.NewsEvent import NewsEventBase
from src.utils.NewsArticle import NewsArticle


# ================================================
# Helper functions
# ================================================


def literal_converter(val):
    try:
        return ast.literal_eval(val)
    except Exception:
        return val


def create_events(df):
    clusterIds = df["clusterId"].unique()

    events = [
        NewsEventBase(
            articles=[
                NewsArticle(a)
                for a in df[df["clusterId"] == clusterId].to_dict("records")
            ]
        )
        for clusterId in clusterIds
    ]
    events = sorted(events, key=lambda e: e.min_time)
    return events


def load_events(input_file):
    df = pd.read_csv(
        input_file,
        names=[
            "id",
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
        dtype={
            "id": "Int64",
            "title": "str",
            "body": "str",
            "lang": "str",
            "source": "str",
            "dateTime": "str",
            "url": "str",
            "uri": "str",
            "eventUri": "str",
            "concepts": "str",
            "clusterId": "str",
            "namedEntities": "str",
            "wikiConcepts": "str",
        },
        parse_dates=["dateTime"],
        on_bad_lines="warn",
        engine="python",
        skiprows=1,
    )
    # dataframe sorting and init
    df.drop(columns=["wikiConcepts", "namedEntities"], inplace=True)
    df.sort_values(by="dateTime", inplace=True)
    events = create_events(df)
    return df, events


def create_dataframe(events, drop_duplicates=False):
    """Store all articles into the dataframe"""

    data = [
        article.to_array()[:-2]
        for event in tqdm(events, desc="Event prep")
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
    df.sort_values(by="dateTime", inplace=True)
    if drop_duplicates:
        df.drop_duplicates(["uri"], keep="first", inplace=True)
    df.reset_index(drop=True, inplace=True)
    df.index.name = "id"
    return df


def assign_new_cluster_ids(events, last_index=1):
    cluster_id = last_index
    for event in events:
        event.assign_cluster_id(f"wn-{cluster_id}")
        cluster_id += 1
    return cluster_id


# ================================================
# Main function
# ================================================


def main(args):
    event_files = [
        f
        for f in listdir(args.manual_eval_dir)
        if isfile(join(args.manual_eval_dir, f))
    ]
    Path(args.merge_file_path).parent.mkdir(parents=True, exist_ok=True)

    all_events = []
    last_index = 1
    for file in tqdm(event_files, desc="Files"):
        _, events = load_events(f"{args.manual_eval_dir}/{file}")
        last_index = assign_new_cluster_ids(events, last_index)
        all_events = all_events + events

    df = create_dataframe(all_events, args.drop_duplicates)
    df.to_csv(args.merge_file_path, encoding="utf-8", index=True)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--manual_eval_dir", default=None, type=str)
    parser.add_argument("--merge_file_path", default=None, type=str)
    parser.add_argument("--drop_duplicates", action="store_true")
    args = parser.parse_args()

    main(args)
