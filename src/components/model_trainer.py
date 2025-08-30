import os
import sys
from dataclasses import dataclass

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.metrics import r2_score
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

from src.exception import CustomException
from src.logger import logger
from src.utils import save_object, evaluate_models


@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join("artifacts", "model.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_training(self, train_array, test_array):
        try:
            logger.info("Splitting training and test input data")

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
                "AdaBoost": AdaBoostRegressor(),
                "XGBoost": XGBRegressor(),
                "CatBoost": CatBoostRegressor(verbose=0),
            }

            params = {
                "Linear Regression": {},
                "Ridge": {"alpha": [0.1, 1.0, 10]},
                "Lasso": {"alpha": [0.1, 1.0, 10]},
                "Decision Tree": {"max_depth": [3, 5, 7]},
                "Random Forest": {"n_estimators": [50, 100]},
                "Gradient Boosting": {"n_estimators": [50, 100]},
                "AdaBoost": {"n_estimators": [50, 100]},
                "XGBoost": {"n_estimators": [50, 100]},
                "CatBoost": {"depth": [6, 8], "learning_rate": [0.01, 0.1]},
            }

            logger.info("Training multiple models with hyperparameter tuning")
            model_report = evaluate_models(X_train, y_train, X_test, y_test, models, params)

            best_model_score = max(model_report.values())
            best_model_name = max(model_report, key=model_report.get)
            best_model = models[best_model_name]

            logger.info(f"Best model found: {best_model_name} with R² score: {best_model_score}")

            if best_model_score < 0.6:
                raise CustomException("No suitable model found with R² > 0.6", sys)

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model,
            )

            logger.info(f"Model saved at {self.model_trainer_config.trained_model_file_path}")

            predicted = best_model.predict(X_test)
            r2 = r2_score(y_test, predicted)

            return r2

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    import numpy as np

    train_array = np.load(os.path.join("artifacts", "train_array.npy"))
    test_array = np.load(os.path.join("artifacts", "test_array.npy"))

    trainer = ModelTrainer()
    r2 = trainer.initiate_model_training(train_array, test_array)
    print("Final R² score on test data:", r2)
