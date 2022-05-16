import pytest
from src.housing import models


@pytest.mark.parametrize("expected", [("installed")])
def test_linear_regression(expected):
    try:
        models.DecisionTreeRegressorHousing()
        models.LinearRegressionModelHouse()
        models.RandomForestRegressorHousing()
        models.SupportVectorRegressionHousing()
        res = "installed"
    except Exception as e:
        res = e

    assert res == expected
