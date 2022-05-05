import os
import unittest

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

rooms_ix, bedrooms_ix, population_ix, households_ix = 3, 4, 5, 6
HOUSING_PATH = "data"


class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
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


def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)


def split(housing, housing_path=HOUSING_PATH):

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


class MyTest(unittest.TestCase):
    def test(self):
        housing = load_housing_data()
        training_set, testing_set = split(housing)
        training_similar = pd.read_csv("data/training_set.csv")
        testing_similar = pd.read_csv("data/test_set.csv")
        self.assertAlmostEquals(np.sum(training_set[1]), np.sum(training_similar["1"]))
