import os
import sys
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting training and test input data")

            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1],
            )

            models = {
                "Linear Regression": LinearRegression(),
                "Ridge": Ridge(),
                "Lasso": Lasso(),
                "Decision Tree": DecisionTreeRegressor(),
                "Random Forest": RandomForestRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "XGBoost": XGBRegressor(),
                "CatBoost": CatBoostRegressor(verbose=0),
            }

            params = {
                "Decision Tree": {
                    "criterion": ["squared_error", "friedman_mse", "absolute_error"]
                },
                "Random Forest": {
                    "n_estimators": [50, 100, 200],
                    "max_depth": [None, 10, 20],
                },
                "Gradient Boosting": {
                    "n_estimators": [50, 100],
                    "learning_rate": [0.01, 0.1, 0.2],
                    "subsample": [0.8, 1.0],
                },
                "XGBoost": {
                    "n_estimators": [50, 100],
                    "learning_rate": [0.01, 0.1, 0.2],
                    "max_depth": [3, 6, 10],
                },
                "CatBoost": {
                    "iterations": [50, 100],
                    "learning_rate": [0.01, 0.1],
                    "depth": [6, 8, 10],
                },
            }

            model_report = {}
            best_model_name = None
            best_model_score = -np.inf
            best_model = None

            # Train & evaluate models
            for model_name, model in models.items():
                logging.info(f"Training {model_name}...")

                if model_name in params:
                    gs = GridSearchCV(model, params[model_name], cv=3, n_jobs=-1, scoring="r2")
                    gs.fit(X_train, y_train)
                    model = gs.best_estimator_
                else:
                    model.fit(X_train, y_train)

                y_train_pred = model.predict(X_train)
                y_test_pred = model.predict(X_test)

                train_score = r2_score(y_train, y_train_pred)
                test_score = r2_score(y_test, y_test_pred)

                model_report[model_name] = test_score

                logging.info(f"{model_name} → Train R2: {train_score:.4f}, Test R2: {test_score:.4f}")

                if test_score > best_model_score:
                    best_model_score = test_score
                    best_model_name = model_name
                    best_model = model

            logging.info(f"Best model: {best_model_name} with R2: {best_model_score:.4f}")

            if best_model_score < 0.6:
                raise CustomException("No suitable model found with R2 >= 0.6")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model,
            )

            logging.info("Best model saved successfully.")

            return best_model_name, best_model_score

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    from src.components.data_transformation import DataTransformation

    obj = DataTransformation()
    train_arr, test_arr, _ = obj.initiate_data_transformation(
        os.path.join("artifacts", "train.csv"),
        os.path.join("artifacts", "test.csv"),
    )

    trainer = ModelTrainer()
    model_name, score = trainer.initiate_model_trainer(train_arr, test_arr)
    print(f"✅ Best model: {model_name} with R2 score: {score:.4f}")
