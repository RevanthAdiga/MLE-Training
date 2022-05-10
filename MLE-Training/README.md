# Median housing value prediction

The housing data can be downloaded from https://raw.githubusercontent.com/ageron/handson-ml/master/. The script has codes to download the data. We have modelled the median house value on given housing data.

The following techniques have been used:

 - Linear regression
 - Decision Tree
 - Random Forest
 - Support Vector Regression

## Steps performed
 - We prepare and clean the data. We check and impute for missing values.
 - Features are generated and the variables are checked for correlation.
 - Multiple sampling techinuqies are evaluated. The data set is split into train and test.
 - All the above said modelling techniques are tried and evaluated. The final metric used to evaluate is mean squared error.


# Folder Structure
    .
    ├── "LICENSE"
    ├── "README.md"
    ├── "artifacts"
    ├── "data"
    ├── "deploy"
    ├── "dist"
    ├── "docs"
    ├── "ingest_data.py"
    ├── "logs"
    ├── "notebooks"
    ├── "pyproject.toml"
    ├── "requirements.txt"
    ├── "score.py"
    ├── "setup.cfg"
    ├── "setup.py"
    ├── "src"
    │   ├── "__init__.py"
    │   ├── "housing"
    │   │   ├── "__init__.py"
    │   │   └── "models.py"
    │   └── "utils"
    │       ├── "__init__.py"
    │       ├── "custom_transformer.py"
    │       └── "fetch_load_data.py"
    ├── "tests"
    │   ├── "test_installation.py"
    │   └── "test_training.py"
    │   └── "test_data_ingestion.py"
    └── "train.py"

### Activate conda env
    run in deploy/conda dir
    - conda env create -f env.yml
    - conda activate mle-dev

## To install packages
    run in root dir
    - python -m pip install -i https://test.pypi.org/simple/ housing_revanth==0.0.1
    or
    for dev-mode use(one) of the following methods:
    1. - python -m build
       - pip install dist/housing-revanth-0.0.1.tar.gz
    2. - python setup.py install
## Verify installation
    run in python prompt
    >> import housing

## setup.cfg
Contains all the configuration for building distributions

## Workflow
By default log is consoled
Default log path is logs/app.log
check command args to know more for script by passing <script.py> -h
### Download data and create train and validation sets
    run in root dir
    - python ingest_data.py
### Train data
    run in root dir
    - python train.py
### Score data
    run in root dir
    - python score.py

## housing package
### modules
 - LinearRegressionModelHouse
 - DecisionTreeRegressorHousing
 - RandomForestRegressorHousing
 - Support Vector Regression

Each module has two methods
 - transform: to transform housing data
 - fit: to train housing data

## tests
    run in root dir
    - pytest tests

#   Docker
    pull latest image
    - $ docker pull revanthadiga3/housing
    run and rename(for better use) in detached mode
    - $ docker run -d -t --name housing revanthadiga3/housing:latest
    start docker container(if not already started)
    - $ docker start housing
    enter exec mode for test/train/score
    - $ docker exec -it housing bash
