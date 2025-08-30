import os
import sys
import dill
import pickle
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from src.exception import CustomException
from src.logger import logging

def save_object(file_path, obj):
    """
    Saves any Python object (model, preprocessor, etc.) using dill.
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

        logging.info(f"Object saved at: {file_path}")

    except Exception as e:
        raise CustomException(e, sys)


def load_object(file_path):
    """
    Loads a Python object (model, preprocessor, etc.) saved with dill.
    """
    try:
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)


def evaluate_models(X_train, y_train, X_test, y_test, models: dict):
    """
    Train and evaluate multiple models.
    Returns a dictionary {model_name: R2_score}.
    """
    try:
        report = {}

        for model_name, model in models.items():
            logging.info(f"Training {model_name}")
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)

            r2 = r2_score(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))

            report[model_name] = r2

            logging.info(f"{model_name} -> R2: {r2:.4f}, MAE: {mae:.4f}, RMSE: {rmse:.4f}")

        return report

    except Exception as e:
        raise CustomException(e, sys)
