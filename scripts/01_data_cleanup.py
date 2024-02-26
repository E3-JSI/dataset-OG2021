import os
import json
import datetime
from tqdm import tqdm
from argparse import ArgumentParser

# import the dataset loader
from src.data.dataset import load_dataset

# ================================================
# Static variables
# ================================================

DIRNAME = os.path.dirname(__file__)

# ================================================
# Helper functions
# ================================================


class NewsArticleEncoder(json.JSONEncoder):
    """ "Used to Serialize a datetime encoder"""

    def default(self, z):
        if isinstance(z, datetime.datetime):
            return str(z)
        else:
            return super().default(z)


DATA_PATH = os.path.join(DIRNAME, "..", "data", "processed")
# create the processed directory if not exists
if not os.path.isdir(DATA_PATH):
    os.makedirs(DATA_PATH)


# ================================================
# Main function
# ================================================


def main(args):
    # load the raw articles
    dataset = load_dataset(args.raw_dir, dataType="raw")
    # filter out the duplicates
    dataset = list(filter(lambda x: not x["isDuplicate"], dataset))

    print("Dataset loaded".ljust(50, ".") + "done!")

    print("Sort dataset".ljust(50, "."), end="", flush=True)
    # sort news articles in cronological order
    dataset.sort(key=lambda x: x["dateTime"])
    print("done!")

    # store the articles in the articles file
    with open(args.results, mode="w", encoding="utf8") as file:
        for article in tqdm(dataset, desc="Saving progress"):
            json.dump(article, file, ensure_ascii=False, cls=NewsArticleEncoder)
            file.write("\n")

    print("Dataset written to JSON".ljust(50, ".") + "done!")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--raw_dir", type=str)
    parser.add_argument("--results", type=str)
    args = parser.parse_args()
    main(args)
