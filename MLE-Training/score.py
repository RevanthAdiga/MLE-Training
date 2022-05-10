import argparse
import logging
import os

import joblib
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.utils.custom_transformer import CombinedAttributesAdder
from src.utils.fetch_load_data import load_housing_data

CombinedAttributesAdder()

HOUSING_PATH = os.path.join("datasets/validation.csv")
MODEL_PATH = os.path.join("artifacts/models")
LOG_PATH = os.path.join("logs/app.log")

parser = argparse.ArgumentParser()
parser.add_argument(
    "-d", "--data", help="directory name to import data from", nargs="?", const=HOUSING_PATH, default=HOUSING_PATH,
)
parser.add_argument(
    "-m",
    "--model",
    help="""Path to save model to (with filename) if not provided will run all
    the models in artifacts/model""",
    nargs="?",
    const=MODEL_PATH,
    default=MODEL_PATH,
)

parser.add_argument(
    "--log-level", help="Define LOG LEVEL", nargs="?", const="DEBUG", default="DEBUG",
)
parser.add_argument("--log-path", help="log to a file", nargs="?", const=LOG_PATH)
parser.add_argument(
    "--no-console-log", help="disable console logging", nargs="?", const=True, type=bool,
)

args = parser.parse_args()
logger = logging.getLogger()

if args.log_level and not args.log_path:
    logging.basicConfig(level=args.log_level)

if args.log_path:
    logging.basicConfig(
        level=args.log_level, filename=args.log_path, filemode="a", format="%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s", datefmt="%H:%M:%S",
    )
    print(args.log_path)

if args.no_console_log:
    logger.disabled = True

logging.debug(f"--data -> {args.data}")

housing = load_housing_data(args.data)
logging.debug(f"--data.shape -> {housing.shape}\n")

housing_score = housing.drop("median_house_value", axis=1)  # drop labels for training set
housing_labels_score = housing["median_house_value"].copy()


def score_models(model):
    logging.debug(f"--model -> {model}")
    model = joblib.load(model)
    housing_predictions = model.predict(housing_score)
    mse = mean_squared_error(housing_labels_score, housing_predictions)
    rmse = np.sqrt(mse)
    print("rmse: ", rmse)
    mae = mean_absolute_error(housing_labels_score, housing_predictions)
    print("mae: ", mae)


if os.path.isdir(args.model):
    for model_ in os.listdir(args.model):
        score_models(os.path.join(args.model, model_))
else:
    score_models(args.model)
