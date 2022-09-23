import os
import json
import pathlib
from datetime import datetime

# static data location
DATA_PATHS = {
    "raw": os.path.join(
        pathlib.Path(__file__).parent.parent.parent.absolute(), "data/raw"
    ),
    "processed": os.path.join(
        pathlib.Path(__file__).parent.parent.parent.absolute(), "data/processed"
    ),
}


def format_article(article, dataType="processed"):
    DATE_FORMAT = "%Y-%m-%d" if dataType == "raw" else "%Y-%m-%d %H:%M:%S"
    DATETIME_FORMAT = "%Y-%m-%dT%H:%M:%SZ" if dataType == "raw" else "%Y-%m-%d %H:%M:%S"

    return {
        **article,
        "date": datetime.strptime(article["date"], DATE_FORMAT)
        if article["date"]
        else None,
        "dateTime": datetime.strptime(article["dateTime"], DATETIME_FORMAT)
        if article["dateTime"]
        else None,
        "dateTimePub": datetime.strptime(article["dateTimePub"], DATETIME_FORMAT)
        if article["dateTimePub"]
        else None,
    }


def load_dataset(fpath: str = DATA_PATHS["processed"], dataType="processed"):
    """Get all of the articles in a single array

    Args:
        fpath (str): The directory path from which we wish to collect the data.
    """
    articles = []
    # iterate through all of the files and folders
    for file in os.listdir(fpath):
        filepath = os.path.join(fpath, file)
        if os.path.isfile(filepath):
            add_attrs = {}
            if dataType == "raw":
                concepts = file.replace(".jsonl", "").split("-")[0]
                add_attrs["concepts"] = concepts.split("&")
            # open the file and retrieve all of the article metadata
            with open(filepath, mode="r", encoding="utf8") as file:
                articles = articles + [
                    {**format_article(json.loads(line), dataType), **add_attrs}
                    for line in file.readlines()
                ]
        else:
            # append the directory articles
            articles = articles + load_dataset(filepath, dataType)

    # return the articles
    return articles
