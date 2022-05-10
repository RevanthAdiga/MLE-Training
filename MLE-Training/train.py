import argparse
import logging
import os

import joblib

from src.housing.models import (
    DecisionTreeRegressorHousing,
    LinearRegressionModelHouse,
    RandomForestRegressorHousing,
    SupportVectorRegressionHousing,
)
from src.utils.fetch_load_data import load_housing_data

HOUSING_PATH = os.path.join("datasets/train.csv")
MODEL_PATH = os.path.join("artifacts/models")
LOG_PATH = os.path.join("logs/app.log")

parser = argparse.ArgumentParser()
parser.add_argument(
    "-d", "--data", help="directory name to import data from", nargs="?", const=HOUSING_PATH, default=HOUSING_PATH,
)
parser.add_argument(
    "-m", "--model", help="Path to save model to (with filename)", nargs="?", const=MODEL_PATH, default=MODEL_PATH,
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

data = load_housing_data(args.data)
logging.debug(f"--data.shape -> {data.shape}\n")

X = data.drop("median_house_value", axis=1)
y = data["median_house_value"].copy()


# linear model
logging.debug("--training -> LinearRegressionModelHouse")
linear_model = LinearRegressionModelHouse()
linear_model_pipeline = linear_model.fit(X, y)
logging.debug(f"--model -> {linear_model_pipeline}")
joblib.dump(linear_model_pipeline, os.path.join(args.model, "Linear_Reg.jlib"))


# Decision tree model
logging.debug("--training -> DecisionTreeRegressorHousing")
decision_model = DecisionTreeRegressorHousing()
decision_model_pipeline = decision_model.fit(X, y)
logging.debug(f"--model -> {decision_model_pipeline}")
joblib.dump(decision_model_pipeline, os.path.join(args.model, "Decision_Reg.jlib"))

# Random forest model
logging.debug("--training -> RandomForestRegressorHousing")
random_model = RandomForestRegressorHousing()
random_model_pipeline = random_model.fit(X, y)
logging.debug(f"--model -> {random_model_pipeline}")
joblib.dump(random_model_pipeline, os.path.join(args.model, "Random_Reg.jlib"))

# Support Vector model
logging.debug("--training -> SupportVectorRegressorHousing")
support_vector_model = SupportVectorRegressionHousing()
support_vector_pipeline = support_vector_model.fit(X, y)
logging.debug(f"--model -> {support_vector_pipeline}")
joblib.dump(support_vector_pipeline, os.path.join(args.model, "SupportVector_Reg.jlib"))
