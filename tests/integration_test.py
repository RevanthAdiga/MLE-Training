import os
import unittest

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def eval_metrics(actual, pred):
    # compute relevant metrics
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2


def score(housing_path="", model="Lin_reg.pkl"):

    test = os.path.join(housing_path, "data/test_set.csv")
    testset = pd.read_csv(test)

    testx = testset.drop("median_house_value", axis=1)
    testy = testset["median_house_value"].copy()

    path = os.path.join(housing_path, "models")
    modelx = os.path.join(path, model)

    reg = joblib.load(modelx)
    predicted_qualities = reg.predict(testx)

    (rmse, mae, r2) = eval_metrics(testy, predicted_qualities)

    return rmse, mae, r2


class MyTest(unittest.TestCase):
    def test(self):
        rmse, mae, r2 = score(model="Lin_Reg.pkl")

        self.assertEqual(rmse, 66844.68513121424)
