import ast
import warnings

from os import listdir
from os.path import isfile, join, exists, getsize

import pandas as pd
from tqdm import tqdm
from pathlib import Path
from argparse import ArgumentParser

from src.utils.NewsEventBase import NewsEvent
from src.utils.MultiNewsEventMonitor import MultiNewsEventMonitor
from src.utils.NewsArticle import NewsArticle

warnings.simplefilter(action="ignore")


def literal_converter(val):
    try:
        return ast.literal_eval(val)
    except Exception:
        return val


def create_dataframe(event_monitor):
    """Store all articles into the dataframe"""

    events = event_monitor.merge_multi_event_clusters()
    data = [
        article.to_array()
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
            "namedEntities",
            "wikiConcepts",
        ],
    )
    return df


def load_events(input_file, run_as_test):
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
            "concepts": "string",
            "clusterId": "str",
            "namedEntities": "str",
            "wikiConcepts": "str",
        },
        parse_dates=["dateTime"],
        on_bad_lines="warn",
        engine="python",
        skiprows=1,
    )

    # Some news dont have titles (end with an error) we drop those news
    df = df[df["title"].notna() & df["title"].notnull()]

    # reset the ID column
    df = df.sort_values(by="dateTime")
    df["concepts"] = df["concepts"].apply(lambda x: literal_converter(x))
    df = df.where(df.notnull() & df.notna(), None)

    cluster_ids = df["clusterId"].unique()
    cluster_ids = cluster_ids[:200] if run_as_test else cluster_ids

    # create the news events list
    events = [
        NewsEvent(
            articles=[
                NewsArticle(a)
                for a in df[df["clusterId"] == cluster_id].to_dict("records")
            ]
        )
        for cluster_id in tqdm(cluster_ids, desc="File load")
    ]
    return events


def cluster_and_save_events(
    input_file,
    output_file,
    sim_threshold,
    time_threshold_in_days,
    w_reg,
    w_nit,
    run_as_test=False,
):
    event_monitor = MultiNewsEventMonitor(
        sim_threshold=sim_threshold,
        time_threshold_in_days=time_threshold_in_days,
        w_reg=w_reg,
        w_nit=w_nit,
    )

    events = load_events(input_file, run_as_test)
    for event in tqdm(events, desc=input_file.split("/")[-1]):
        # specify where we compare the articles
        event_monitor.update(event)

    df = create_dataframe(event_monitor)
    df.to_csv(output_file, encoding="utf-8", index=True)


def main(args):
    mono_event_files = sorted(
        [
            f
            for f in listdir(args.mono_events_dir)
            if isfile(join(args.mono_events_dir, f))
        ],
        key=lambda file: getsize(f"{args.mono_events_dir}/{file}"),
    )

    # create the results directory
    Path(args.multi_events_dir).mkdir(parents=True, exist_ok=True)
    for file in tqdm(mono_event_files, desc="Files"):
        if exists(f"{args.multi_events_dir}/{file}"):
            continue
        cluster_and_save_events(
            input_file=f"{args.mono_events_dir}/{file}",
            output_file=f"{args.multi_events_dir}/{file}",
            sim_threshold=args.sim_threshold,
            time_threshold_in_days=args.time_threshold_in_days,
            w_reg=args.w_reg,
            w_nit=args.w_nit,
            run_as_test=args.test,
        )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--mono_events_dir", default=None, type=str)
    parser.add_argument("--multi_events_dir", default=None, type=str)
    parser.add_argument("--sim_threshold", default=0.93, type=float)
    parser.add_argument("--time_threshold_in_days", default=1, type=int)
    parser.add_argument("--w_reg", default=0.1, type=float)
    parser.add_argument("--w_nit", default=100, type=int)
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()

    main(args)
