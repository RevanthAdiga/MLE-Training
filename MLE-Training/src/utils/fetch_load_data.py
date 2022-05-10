import os
import tarfile
import urllib.request

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit


def fetch_housing_data(housing_url, housing_path):
    try:
        os.makedirs(housing_path, exist_ok=True)
        tgz_path = os.path.join(housing_path, "housing.tgz")
        urllib.request.urlretrieve(housing_url, tgz_path)
        housing_tgz = tarfile.open(tgz_path)
        housing_tgz.extractall(path=housing_path)
        housing_tgz.close()
        return "data fetched"
    except Exception as e:
        return e


def split_housing_data(housing, housing_path):
    housing["income_cat"] = pd.cut(housing["median_income"], bins=[0.0, 1.5, 3.0, 4.5, 6.0, np.inf], labels=[1, 2, 3, 4, 5],)
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, test_index in split.split(housing, housing["income_cat"]):
        strat_train_set = housing.loc[train_index]
        strat_test_set = housing.loc[test_index]

    strat_train_set.to_csv(os.path.join(housing_path, "train.csv"))
    strat_test_set.to_csv(os.path.join(housing_path, "validation.csv"))

    return (strat_train_set, strat_test_set)


def load_housing_data(housing_path):
    try:
        csv_path = os.path.join(housing_path)
        return pd.read_csv(csv_path)
    except Exception as e:
        return e
