import os

import pandas as pd
import pytest
from src.utils.fetch_load_data import (
    fetch_housing_data,
    load_housing_data,
    split_housing_data,
)

DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
HOUSING_PATH = os.path.join("datasets/")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"


@pytest.mark.parametrize(
    "test_input, expected",
    [
        (
            {"housing_url": HOUSING_URL, "housing_path": HOUSING_PATH},
            "data fetched",
        )
    ],
)
def test_fetch_housing_data(test_input, expected):
    assert (
        fetch_housing_data(
            housing_url=test_input["housing_url"],
            housing_path=test_input["housing_path"],
        )
        == expected
    )


@pytest.mark.parametrize(
    "test_input, expected",
    [(os.path.join(HOUSING_PATH, "housing.csv"), type(pd.DataFrame()))],
)
def test_load_housing_data(test_input, expected):
    assert type(load_housing_data(test_input)) == expected


@pytest.mark.parametrize(
    "test_input, expected",
    [
        (
            {
                "housing_path": HOUSING_PATH,
                "housing": os.path.join(HOUSING_PATH, "housing.csv"),
            },
            (type(pd.DataFrame()), type(pd.DataFrame())),
        )
    ],
)
def test_split_housing_data(test_input, expected):
    res = split_housing_data(
        housing=load_housing_data(housing_path=test_input["housing"]),
        housing_path=test_input["housing_path"],
    )
    assert (type(res[0]), type(res[1])) == expected
