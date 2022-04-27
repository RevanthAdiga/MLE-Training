"""Training of Datasets

This module demonstates training the models based on training datasets and results models
that can used for predictions and evaluation of models


Syntax to run the file
----------------------
    $ python src/train.py "Enter the common directory to save the regression models"

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
import warnings

import joblib
import pandas as pd
import yaml
from logging_tree import printout
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor


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
    """Reads the configured path ie., env.yaml file to get the training preprocessed datasets

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
    data_path = config["train"]["deps"]
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


def training(housing_path="", model=LinearRegression(), save_model="Lin_Reg.pkl"):
    """Function to result the specific model training required on training datasets

    Parameters
    ----------
    housing_path : str
    model : ML model
    save_model : str

    Returns
    -------
    saved_model
        Results in storing the trained model in respective model directory

    """
    models = os.path.join(housing_path, "models")
    os.makedirs(models, exist_ok=True)

    train = os.path.join(housing_path, "datasets/training_set.csv")
    trainset = pd.read_csv(train)

    trainx = trainset.drop("median_house_value", axis=1)
    trainy = trainset["median_house_value"].copy()

    model.fit(trainx, trainy)
    print(model.score(trainx, trainy))
    saved_model = os.path.join(models, save_model)
    joblib.dump(model, saved_model)


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

if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="Enter the folder path", default="env.yaml")
    args = parser.parse_args()

    housing_path = get_folder_path(config_path=args.config)

    logger = configure_logger(log_file=r"history.log")
    logger.info("Logging Test - Start")

    training(housing_path, model=LinearRegression(), save_model="Lin_Reg.pkl")
    training(
        housing_path, model=DecisionTreeRegressor(), save_model="Decision_tree.pkl"
    )
    training(
        housing_path,
        model=RandomForestRegressor(max_features=8, n_estimators=30),
        save_model="Random_Forest.pkl",
    )
    training(
        housing_path,
        model=SVR(C=157055.10989448498, kernel="rbf", gamma=0.26497040005002437),
        save_model="SVR.pkl",
    )

    logger.info("Logging Test - Test 2 Done")
    logger.warning("Watch out!")

    printout()
