import os
import json
import datetime
from tqdm import tqdm
# import the dataset loader
from src.data.dataset import load_dataset, DATA_PATHS

#================================================
# Static variables
#================================================

DIRNAME = os.path.dirname(__file__)

#================================================
# Helper functions
#================================================

class NewsArticleEncoder(json.JSONEncoder):
    """"Used to Serialize a datetime encoder"""
    def default(self, z):
        if isinstance(z, datetime.datetime):
            return (str(z))
        else:
            return super().default(z)


DATA_PATH = os.path.join(DIRNAME, "..", "data", "processed")
# create the processed directory if not exists
if not os.path.isdir(DATA_PATH):
    os.makedirs(DATA_PATH)


#================================================
# Main function
#================================================

def main():
    # load the raw articles
    dataset = load_dataset(DATA_PATHS["raw"], dataType="raw")

    # filter out the duplicates
    dataset = filter(lambda x: not x["isDuplicate"], dataset)
    dataset = list(dataset)

    # sort news articles in cronological order
    dataset.sort(key = lambda x: x["dateTime"])

    # store the articles in the articles file
    with open(os.path.join(DATA_PATH, "articles.jsonl"), mode="w", encoding="utf8") as file:
        for article in tqdm(dataset, desc="Saving progress"):
            json.dump(article, file, ensure_ascii=False, cls=NewsArticleEncoder)
            file.write("\n")


if __name__ == "__main__":
    main()
