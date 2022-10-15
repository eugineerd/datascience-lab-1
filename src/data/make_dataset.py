# -*- coding: utf-8 -*-
import click
import logging
import os
import sys
import pandas as pd
import pickle
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
from sklearn.model_selection import train_test_split
from src.data.preprocess import preprocess


@click.command()
@click.argument("input_filepath", type=click.Path(exists=True))
@click.argument("output_filepath", type=click.Path())
def main(input_filepath: str, output_filepath: str):
    """Runs data processing scripts to turn raw data from (../raw) into
    cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info("making final data set from raw data")

    pickle_train(input_filepath, output_filepath)
    pickle_test(input_filepath, output_filepath)


def pickle_test(input_filepath: str, output_filepath: str):
    path_X = os.path.join(input_filepath, "test_dataset_test.csv")
    path_y = os.path.join(input_filepath, "sample_solution.csv")
    df_X = pd.read_csv(path_X)
    df_y = pd.read_csv(path_y)
    df_y.rename({"ID": "ID_y"}, axis=1, inplace=True)
    df = pd.concat([df_X, df_y], axis=1)
    df = preprocess(df)
    pickle_path = os.path.join(output_filepath, "test.pkl")
    with open(pickle_path, "wb") as f:
        pickle.dump(df, f)


def pickle_train(input_filepath: str, output_filepath: str):
    file = "train.csv"
    path = os.path.join(input_filepath, file)
    df = pd.read_csv(path)
    df = preprocess(df)
    pickle_name = file.rsplit(".", 1)[0] + ".pkl"
    pickle_path = os.path.join(output_filepath, pickle_name)
    with open(pickle_path, "wb") as f:
        pickle.dump(df, f)


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
