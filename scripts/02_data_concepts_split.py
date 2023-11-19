import pandas as pd
from tqdm import tqdm
from pathlib import Path
from argparse import ArgumentParser

# import the dataset loader
from src.data.dataset import load_dataset


def main(args):
    # load the articles
    dataset = load_dataset(args.articles_dir, dataType="processed")
    dataset = pd.DataFrame(dataset)

    # list of all unique concepts
    unique_concepts = dataset["concepts"].drop_duplicates().to_list()

    # create the concept directory
    Path(args.concepts_dir).mkdir(parents=True, exist_ok=True)

    # iterate through the concepts and create the csvs
    for concepts in tqdm(unique_concepts, desc="concepts"):
        # get the mask IDs of the articles containing the concepts
        mask = dataset["concepts"].apply(lambda x: all(item in x for item in concepts))
        c_df = dataset[mask]

        name = "__".join([str(y) for y in concepts])
        file_path = f"{args.concepts_dir}/{name}.csv"
        c_df.to_csv(file_path, encoding="utf-8", index=True)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--articles_dir", type=str)
    parser.add_argument("--concepts_dir", type=str)
    args = parser.parse_args()
    main(args)
