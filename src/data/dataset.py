import os
import json
import pathlib
from datetime import datetime

# static data location

RAW_DATA_PATH = os.path.join(
    pathlib.Path(__file__).parent.parent.parent.absolute(), "data/raw"
)

DATE_FORMAT = "%Y-%m-%d"
DATETIME_FORMAT = "%Y-%m-%dT%H:%M:%SZ"


def format_article(article):

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


def load_dataset(fpath: str = RAW_DATA_PATH):
    """Get all of the articles in a single array

    Args:
        fpath (str): The directory path from which we wish to collect the data.
    """

    articles = []
    # iterate through all of the files and folders
    for file in os.listdir(fpath):
        filepath = os.path.join(fpath, file)
        if os.path.isfile(filepath):
            [concepts, _, _] = file.split(".")[0].split("-")
            add_attrs = {
                "concepts": concepts.split("&"),
            }
            # open the file and retrieve all of the article metadata
            with open(filepath, mode="r", encoding="utf8") as file:
                articles = articles + [
                    {**format_article(json.loads(line)), **add_attrs}
                    for line in file.readlines()
                ]
        else:
            # append the directory articles
            articles = articles + load_dataset(filepath)

    # return the articles
    return articles
