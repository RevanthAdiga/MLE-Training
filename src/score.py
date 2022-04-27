"""Testing and Evaluation of Datasets

This module demonstates testing the models based on testing datasets and resulted models
so that it can used for predictions and evaluation of models


Syntax to run the file
----------------------
    $ python src/score.py "Enter the common directory to load the regression models"

Notes
-----
    Please  note the following librabries need to be installed before you run this module.
    To install the librabries, run this commmand

        $ pip install -r requirements.txt

Attributes
----------

HOUSING_PATH : str
    Defines the folder in which datasets are extracted
"""
import argparse
import logging
import logging.config
import os

import joblib
import numpy as np
import pandas as pd
import yaml
from logging_tree import printout
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

LOGGING_DEFAULT_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S",
        },
        "simple": {"format": "%(message)s"},
    },
    "root": {"level": "DEBUG"},
}


def configure_logger(
    logger=None, cfg=None, log_file=None, console=True, log_level="DEBUG"
):
    """Reads the configured file to save the logs

    Parameters
    ----------
    log_file : file
        The first parameter.

    Returns
    -------
    string
        The datalogs of the program to be stored in log file

    """
    if not cfg:
        logging.config.dictConfig(LOGGING_DEFAULT_CONFIG)
    else:
        logging.config.dictConfig(cfg)

    logger = logger or logging.getLogger()

    if log_file or console:
        for hdlr in logger.handlers:
            logger.removeHandler(hdlr)

        if log_file:
            fh = logging.FileHandler(log_file)
            fh.setLevel(getattr(logging, log_level))
            logger.addHandler(fh)

        if console:
            sh = logging.StreamHandler()
            sh.setLevel(getattr(logging, log_level))
            logger.addHandler(sh)

    return logger


def get_folder_path(config_path):
    """Reads the configured path ie., env.yaml file to get the testing preprocessed datasets

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
    data_path = config["score"]["deps"]
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


def eval_metrics(actual, pred):
    """Function to evaluate based on predicted and original values.

    Parameters
    ----------
    actual : list
    pred : list

    Returns
    -------
    rmse : float
    mae : float
    r2 : float
        Results in metrics evaluation of the lists

    """
    # compute relevant metrics
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2


def score(housing_path, model="Lin_reg.pkl"):
    """Function to result the predicted outputs on testing datasets and evaluate the respective models

    Parameters
    ----------
    housing_path : str
    model : ML model

    Returns
    -------
    rmse : float
    mae : float
    r2 : float
        Results in metrics evaluation of the saved models based on testing_datasets

    """
    test = os.path.join(housing_path, "datasets/test_set.csv")
    testset = pd.read_csv(test)

    testx = testset.drop("median_house_value", axis=1)
    testy = testset["median_house_value"].copy()

    path = os.path.join(housing_path, "models")
    modelx = os.path.join(path, model)

    reg = joblib.load(modelx)
    predicted_qualities = reg.predict(testx)

    (rmse, mae, r2) = eval_metrics(testy, predicted_qualities)

    print(model)
    print("  RMSE: %s" % rmse)
    print("  MAE: %s" % mae)
    print("  R2: %s" % r2)
    print("")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="Enter the folder path", default="env.yaml")
    args = parser.parse_args()

    housing_path = get_folder_path(config_path=args.config)

    logger = configure_logger(log_file=r"history.log")
    logger.info("Logging Test - Start")

    score(housing_path, model="Lin_Reg.pkl")
    score(housing_path, model="Decision_tree.pkl")
    score(housing_path, model="Random_Forest.pkl")
    score(housing_path, model="SVR.pkl")

    logger.info("Logging Test - Test 3 Done")
    logger.warning("Watch out!")

    printout()
