# Use the terminal or an Anaconda Prompt for the following steps:

Create the environment from the env.yml file:
 - conda env create -f env.yml
 - conda activate env

# Median housing value prediction

The housing data can be downloaded from https://raw.githubusercontent.com/ageron/handson-ml/master/. The script has codes to download the data. We have modelled the median house value on given housing data.

The following techniques have been used:

 - Linear regression
 - Decision Tree
 - Random Forest
 - SVR

## Steps performed
 - We prepare and clean the data. We check and impute for missing values.
 - Features are generated and the variables are checked for correlation.
 - Multiple sampling techinuqies are evaluated. The data set is split into train and test.
 - All the above said modelling techniques are tried and evaluated. The final metric used to evaluate is mean squared error.

## To excute the script

Created a req file install the req

 - pip install -r requirements.txt

Open git bash in the file
 - git init
 - git add .
 - git commit -m "first commit"

oneliner updates for readme

 - git add . && git commit -m "update Readme.md"
 - git remote add origin "Your git clone ssh link"
 - git branch -M main
 - git push origin main

tox command -
 - tox

for rebuilding -
 - tox -r

pytest command
 - pytest -v

setup commands -
 - pip install -e .

build your own package commands-
 - python setup.py sdist bdist_wheel

from src folder
 - python src/ingest_data.py "Common dataset directory"
 - python src/train.py "Common dataset directory"

from test folder
 - python src/score.py "Common dataset directory"