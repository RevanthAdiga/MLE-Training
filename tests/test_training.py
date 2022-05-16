import os

import pytest
from src.housing.models import (
    DecisionTreeRegressorHousing,
    LinearRegressionModelHouse,
    RandomForestRegressorHousing,
    SupportVectorRegressionHousing,
)
from src.utils.fetch_load_data import load_housing_data

HOUSING_PATH = os.path.join("datasets/train.csv")
data = load_housing_data(HOUSING_PATH)

X = data.drop("median_house_value", axis=1)
y = data["median_house_value"].copy()


@pytest.mark.parametrize(
    "test_input, expected", [({"X": X, "y": y}, "<class 'sklearn.pipeline.Pipeline'>")],
)
def test_linear_regression(test_input, expected):

    model = LinearRegressionModelHouse()
    model_pipeline = model.fit(test_input["X"], test_input["y"])
    assert str(type(model_pipeline)) == expected


@pytest.mark.parametrize(
    "test_input, expected", [({"X": X, "y": y}, "<class 'sklearn.pipeline.Pipeline'>")],
)
def test_decision_tree_regression(test_input, expected):

    model = DecisionTreeRegressorHousing()
    model_pipeline = model.fit(test_input["X"], test_input["y"])
    assert str(type(model_pipeline)) == expected


@pytest.mark.parametrize(
    "test_input, expected", [({"X": X, "y": y}, "<class 'sklearn.pipeline.Pipeline'>")],
)
def test_random_forest_regressor(test_input, expected):
    model = RandomForestRegressorHousing()
    model_pipeline = model.fit(test_input["X"], test_input["y"])
    assert str(type(model_pipeline)) == expected


@pytest.mark.parametrize(
    "test_input, expected", [({"X": X, "y": y}, "<class 'sklearn.pipeline.Pipeline'>")],
)
def test_support_vector_regressor(test_input, expected):
    model = SupportVectorRegressionHousing()
    model_pipeline = model.fit(test_input["X"], test_input["y"])
    assert str(type(model_pipeline)) == expected
