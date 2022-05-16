import argparse
import logging
import os

from src.utils.fetch_load_data import (
    fetch_housing_data,
    load_housing_data,
    split_housing_data,
)

DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
HOUSING_PATH = os.path.join("datasets/")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"
LOG_PATH = os.path.join("logs/app.log")


parser = argparse.ArgumentParser()
parser.add_argument(
    "-d", "--dir", help="Enter the directory name", nargs="?", const=HOUSING_PATH, default=HOUSING_PATH,
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

fetch_housing_data(HOUSING_URL, args.dir)
logging.debug(f"--data path -> {args.dir}")

housing = load_housing_data(os.path.join(args.dir, "housing.csv"))
logging.debug(f"--data.shape -> {housing.shape}")


train_set, test_set = split_housing_data(housing, os.path.join("datasets/"))

logging.debug(f"--train_set.shape -> {train_set.shape}")
logging.debug(f"--test_set.shape -> {test_set.shape}")
