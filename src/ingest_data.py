"""Preparation of Datasets

This module demonstates loading of dataset from a github repository ie.,
`https://raw.githubusercontent.com/ageron/handson-ml/master/datasets/housing/housing.tgz`
and preprocess the datasets for training and testing sets.


Syntax to run the file
----------------------
    $ python src/ingest_data.py

Notes
-----
    Please  note the following librabries need to be installed before you run this module.
    To install the librabries, run this commmand

        $ pip install -r requirements.txt

Attributes
----------
HOUSING_URL : str
    Defines the source for the dataset

HOUSING_PATH : str
    Defines the folder in which datasets are extracted
"""
import argparse
import os
import tarfile
import urllib.request
import warnings

import numpy as np
import pandas as pd
import yaml
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"
HOUSING_PATH = "datasets"
rooms_ix, bedrooms_ix, population_ix, households_ix = 3, 4, 5, 6


def get_data_path(config_path):
    """Reads the configured path ie., env.yaml file to get the source datasets

    Parameters
    ----------
    config_path : str
        The first parameter.

    Returns
    -------
    string
        The default directory mentioned to read the datasets.

    """
    config = read(config_path)
    # print(config)
    data_path = config["load_and_split_data"]["deps"]
    return data_path


def read(config_path):
    """Reads the configured path ie., env.yaml file to get the data

    Parameters
    ----------
    config_path : str
        The first parameter.

    Returns
    -------
    string
        The data inside the yaml file

    """
    with open(config_path) as yaml_file:
        config = yaml.safe_load(yaml_file)
    return config


class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    """Class creates customised attributes to add to the datasets

    Parameters
    ----------
    BaseEstimator : default
    TransformerMixin : default

    Returns
    -------
    np array
        To add to the preprocessed datasets

    """

    def __init__(self, add_bedrooms_per_room=True):
        self.add_bedrooms_per_room = add_bedrooms_per_room

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        rooms_per_household = X[:, rooms_ix] / X[:, households_ix]
        population_per_household = X[:, population_ix] / X[:, households_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[
                X, rooms_per_household, population_per_household, bedrooms_per_room
            ]
        else:
            return np.c_[X, rooms_per_household, population_per_household]


def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    """Function to extract the original datasets

    Parameters
    ----------
    housing_url : str
    housing_path : str

    Returns
    -------
    csv file
        csv file dowloaded at the specific housing_path

    """
    os.makedirs(housing_path, exist_ok=True)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()


def load_housing_data(housing_path=HOUSING_PATH):
    """Function to read the original datasets

    Parameters
    ----------
    housing_path : str

    Returns
    -------
    pd DataFrame
        DataFrame of the csv file is loaded

    """
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)


def split(housing, housing_path=HOUSING_PATH):

    """Function to preprocess the original datasets to obtain a scaled numeric train and test dataframes

    Note
    ----
        Please do read read about handling the categorical data using onehotecoder.
        Attention Required on the usuage of Pipelines and CustomTransformers.

    Parameters
    ----------
    housing : pd DataFrame
    housing_path : str

    Returns
    -------
    csv files
        csv trainingset_file saved at the specific housing_path
        csv testset_file saved at the specific housing_path

    """

    housing["income_cat"] = pd.cut(
        housing["median_income"],
        bins=[0.0, 1.5, 3.0, 4.5, 6.0, np.inf],
        labels=[1, 2, 3, 4, 5],
    )

    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

    for train_index, test_index in split.split(housing, housing["income_cat"]):
        strat_train_set = housing.loc[train_index]
        strat_test_set = housing.loc[test_index]

    for set_ in (strat_train_set, strat_test_set):
        set_.drop("income_cat", axis=1, inplace=True)

    housing = strat_train_set.drop("median_house_value", axis=1)
    housing_labels = strat_train_set["median_house_value"].copy()

    attr_adder = CombinedAttributesAdder(add_bedrooms_per_room=True)

    col_names = "total_rooms", "total_bedrooms", "population", "households"
    rooms_ix, bedrooms_ix, population_ix, households_ix = [
        housing.columns.get_loc(c) for c in col_names
    ]
    housing_extra_attribs = pd.DataFrame(
        attr_adder.transform(housing.values),
        columns=list(housing.columns)
        + ["rooms_per_household", "population_per_household", "bedrooms_per_room"],
        index=housing.index,
    )

    housing = housing_extra_attribs

    housing_num = housing.drop("ocean_proximity", axis=1)

    num_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("attribs_adder", CombinedAttributesAdder()),
            ("std_scaler", StandardScaler()),
        ]
    )

    num_attribs = list(housing_num)
    cat_attribs = ["ocean_proximity"]

    full_pipeline = ColumnTransformer(
        [("num", num_pipeline, num_attribs), ("cat", OneHotEncoder(), cat_attribs)]
    )

    housing_prepared = full_pipeline.fit_transform(housing)

    housing_test = strat_test_set.drop("median_house_value", axis=1)
    housing_test_labels = strat_test_set["median_house_value"].copy()

    attr_adder = CombinedAttributesAdder(add_bedrooms_per_room=True)
    housing_test_extra_attribs = attr_adder.transform(housing_test.values)

    col_names = "total_rooms", "total_bedrooms", "population", "households"
    rooms_ix, bedrooms_ix, population_ix, households_ix = [
        housing_test.columns.get_loc(c) for c in col_names
    ]
    housing_test_extra_attribs = pd.DataFrame(
        housing_test_extra_attribs,
        columns=list(housing_test.columns)
        + ["rooms_per_household", "population_per_household", "bedrooms_per_room"],
        index=housing_test.index,
    )

    housing_test = housing_test_extra_attribs

    housing_test_num = housing_test.drop("ocean_proximity", axis=1)

    num_test_attribs = list(housing_test_num)
    cat_test_attribs = ["ocean_proximity"]

    full_pipeline = ColumnTransformer(
        [
            ("num", num_pipeline, num_test_attribs),
            ("cat", OneHotEncoder(), cat_test_attribs),
        ]
    )

    housing_test_prepared = full_pipeline.fit_transform(housing_test)

    training_set = pd.DataFrame(housing_prepared, housing_labels)
    testing_set = pd.DataFrame(housing_test_prepared, housing_test_labels)

    training_set.reset_index(inplace=True)
    testing_set.reset_index(inplace=True)

    return training_set, testing_set


if __name__ == "__main__":

    warnings.filterwarnings("ignore")
    np.random.seed(42)

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="Enter the folder path", default="env.yaml")
    args = parser.parse_args()

    housing_path = get_data_path(config_path=args.config)
    fetch_housing_data()
    housing = load_housing_data()
    training_set, testing_set = split(housing)

    trainpath = os.path.join(housing_path, "training_set.csv")
    testpath = os.path.join(housing_path, "test_set.csv")

    training_set.to_csv(trainpath)
    testing_set.to_csv(testpath)
