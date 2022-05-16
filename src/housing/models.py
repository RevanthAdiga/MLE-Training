import logging
from random import random
from statistics import median

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from src.utils.custom_transformer import CombinedAttributesAdder


class LinearRegressionModelHouse:
    """
    Model for training house dataset with LinearModel
    """

    def __init__(self):
        self.model_pipeline = None

    def transform(self, X, ret=True):
        """transform X_train/X_test to requirement of model

        Parameters
        ----------
        X : csv
            X_train/X_test
        ret : bool
            True(default) will fit transform_pipeline, False otherwise.
        Returns
        -------
        X_transformed : np.array
        """
        self.X_num = X.drop("ocean_proximity", axis=1)
        logging.warning("Static idx's in CombinedAttributesAdder")
        self.num_pipeline = Pipeline([("imputer", SimpleImputer(strategy="median")), ("attribs_adder", CombinedAttributesAdder()),])

        num_attribs = list(self.X_num)
        cat_attribs = ["ocean_proximity"]

        self.transform_pipeline = ColumnTransformer([("num", self.num_pipeline, num_attribs), ("cat", OneHotEncoder(), cat_attribs),])
        if ret:
            return self.transform_pipeline.fit_transform(X)

    def fit(self, X, y):
        """fits the model with specified inputs

        Parameters
        ----------
        X : csv
            X_train/X_test
        y : csv
            y_train/y_test
        Returns
        -------
        model_pipeline
        """
        self.transform(X, ret=False)
        self.model_pipeline = Pipeline([("transform", self.transform_pipeline), ("model", LinearRegression()),])
        self.model_pipeline.fit(X, y)
        return self.model_pipeline


class DecisionTreeRegressorHousing:
    """
    Model for training house dataset with DecisionTree Regression
    """

    def __init__(self):
        self.model_pipeline = None

    def transform(self, X, ret=True):
        """transform X_train/X_test to requirement of model

        Parameters
        ----------
        X : csv
            X_train/X_test
        ret : bool
            True(default) will fit transform_pipeline, False otherwise.
        Returns
        -------
        X_transformed : np.array
        """
        self.X_num = X.drop("ocean_proximity", axis=1)
        logging.warning("Static idx's in CombinedAttributesAdder")
        self.num_pipeline = Pipeline([("imputer", SimpleImputer(strategy="median")), ("attribs_adder", CombinedAttributesAdder()),])

        num_attribs = list(self.X_num)
        cat_attribs = ["ocean_proximity"]

        self.transform_pipeline = ColumnTransformer([("num", self.num_pipeline, num_attribs), ("cat", OneHotEncoder(), cat_attribs),])
        if ret:
            return self.transform_pipeline.fit_transform(X)

    def fit(self, X, y):
        """fits the model with specified inputs

        Parameters
        ----------
        X : csv
            X_train/X_test
        y : csv
            y_train/y_test
        Returns
        -------
        model_pipeline
        """
        self.transform(X, ret=False)
        self.model_pipeline = Pipeline([("transform", self.transform_pipeline), ("model", DecisionTreeRegressor(random_state=42)),])
        self.model_pipeline.fit(X, y)
        return self.model_pipeline


class RandomForestRegressorHousing:
    """
    Model for training house dataset with RandomForest Regression
    """

    def __init__(self):
        self.model_pipeline = None

    def transform(self, X, ret=True):
        """transform X_train/X_test to requirement of model

        Parameters
        ----------
        X : csv
            X_train/X_test
        ret : bool
            True(default) will fit transform_pipeline, False otherwise.
        Returns
        -------
        X_transformed : np.array
        """
        self.X_num = X.drop("ocean_proximity", axis=1)
        logging.warning("Static idx's in CombinedAttributesAdder")
        self.num_pipeline = Pipeline([("imputer", SimpleImputer(strategy="median")), ("attribs_adder", CombinedAttributesAdder()),])

        num_attribs = list(self.X_num)
        cat_attribs = ["ocean_proximity"]

        self.transform_pipeline = ColumnTransformer([("num", self.num_pipeline, num_attribs), ("cat", OneHotEncoder(), cat_attribs),])
        if ret:
            return self.transform_pipeline.fit_transform(X)

    def fit(self, X, y):
        """fits the model with specified inputs

        Parameters
        ----------
        X : csv
            X_train/X_test
        y : csv
            y_train/y_test
        Returns
        -------
        model_pipeline
        """
        self.transform(X, ret=False)
        self.model_pipeline = Pipeline([("transform", self.transform_pipeline), ("model", RandomForestRegressor(random_state=42, max_features=4, n_estimators=30),),])
        self.model_pipeline.fit(X, y)
        return self.model_pipeline


class SupportVectorRegressionHousing:
    """
    Model for training house dataset with SupportVector Regression
    """

    def __init__(self):
        self.model_pipeline = None

    def transform(self, X, ret):
        """transform X_train/X_test to requirement of model

        Parameters
        ----------
        X : csv
            X_train/X_test
        ret : bool
            True(default) will fit transform_pipeline, False otherwise.
        Returns
        -------
        X_transformed : np.array
        """
        self.X_num = X.drop("ocean_proximity", axis=1)
        logging.warning("Static idx's in CombinedAttributesAdder")
        self.num_pipeline = Pipeline([("imputer", SimpleImputer(strategy="median")), ("attribs_adder", CombinedAttributesAdder()),])

        num_attribs = list(self.X_num)
        cat_attribs = ["ocean_proximity"]

        self.transform_pipeline = ColumnTransformer([("num", self.num_pipeline, num_attribs), ("cat", OneHotEncoder(), cat_attribs),])

        if ret:
            return self.transform_pipeline.fit_transform(X)

    def fit(self, X, y):
        """fits the model with specified inputs

        Parameters
        ----------
        X : csv
            X_train/X_test
        y : csv
            y_train/y_test
        Returns
        -------
        model_pipeline
        """
        self.transform(X, ret=False)
        self.model_pipeline = Pipeline([("transform", self.transform_pipeline), ("model", SVR(C=157055.10989448498, kernel="rbf", gamma=0.26497040005002437,),),])

        self.model_pipeline.fit(X, y)
        return self.model_pipeline

