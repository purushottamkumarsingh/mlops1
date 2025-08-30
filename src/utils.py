import os
import sys
import pickle
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score

from src.exception import CustomException
from src.logger import logger


def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

        logger.info(f"Object saved successfully at {file_path}")

    except Exception as e:
        raise CustomException(e, sys)


def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        raise CustomException(e, sys)


def evaluate_models(X_train, y_train, X_test, y_test, models, params):
    try:
        report = {}

        for model_name, model in models.items():
            logger.info(f"Training {model_name} with GridSearchCV")

            param_grid = params.get(model_name, {})

            gs = GridSearchCV(model, param_grid, cv=3, scoring="r2", n_jobs=-1, verbose=0)
            gs.fit(X_train, y_train)

            best_model = gs.best_estimator_
            best_model.fit(X_train, y_train)

            y_pred = best_model.predict(X_test)
            score = r2_score(y_test, y_pred)

            report[model_name] = score
            logger.info(f"{model_name}: R² Score = {score}")

            models[model_name] = best_model

        return report

    except Exception as e:
        raise CustomException(e, sys)
