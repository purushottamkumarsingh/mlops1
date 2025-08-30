import sys
import os
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from src.exception import CustomException
from src.logger import logging
import pickle


class DataTransformation:
    def __init__(self):
        self.preprocessor_obj_file_path = os.path.join("artifacts", "preprocessor.pkl")

    def get_data_transformer_object(self):
        """
        Creates and returns a ColumnTransformer for preprocessing
        (numerical: imputation + scaling, categorical: imputation + one-hot + scaling).
        """
        try:
            logging.info("Getting data transformer object...")

            # Actual columns from stud.csv
            numerical_columns = ["reading score", "writing score"]
            categorical_columns = [
                "gender",
                "race/ethnicity",
                "parental level of education",
                "lunch",
                "test preparation course"
            ]
            

            # Pipeline for numerical features
            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ]
            )

            # Pipeline for categorical features
            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder", OneHotEncoder(handle_unknown="ignore")),
                    ("scaler", StandardScaler(with_mean=False))
                ]
            )

            # Combine pipelines
            preprocessor = ColumnTransformer(
                transformers=[
                    ("num_pipeline", num_pipeline, numerical_columns),
                    ("cat_pipeline", cat_pipeline, categorical_columns)
                ]
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        """
        Reads train/test data, applies preprocessing, saves preprocessor.
        Returns transformed arrays and the preprocessor file path.
        """
        try:
            logging.info("Initiating data transformation...")

            # Load train and test data
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Train and test data loaded successfully.")
            logging.info(f"Train columns: {list(train_df.columns)}")
            logging.info(f"Test columns: {list(test_df.columns)}")

            # Target column
            target_column = "math score"

            # Separate input and target features
            input_feature_train_df = train_df.drop(columns=[target_column], axis=1)
            target_feature_train_df = train_df[target_column]

            input_feature_test_df = test_df.drop(columns=[target_column], axis=1)
            target_feature_test_df = test_df[target_column]

            # Get preprocessing object
            preprocessor = self.get_data_transformer_object()

            # Transform the data
            logging.info("Applying preprocessing to train and test data.")
            input_feature_train_arr = preprocessor.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessor.transform(input_feature_test_df)

            # Combine transformed features with target
            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[
                input_feature_test_arr, np.array(target_feature_test_df)
            ]

            # Save preprocessor object
            os.makedirs(os.path.dirname(self.preprocessor_obj_file_path), exist_ok=True)
            with open(self.preprocessor_obj_file_path, "wb") as f:
                pickle.dump(preprocessor, f)

            logging.info("Preprocessor object saved successfully.")

            return (
                train_arr,
                test_arr,
                self.preprocessor_obj_file_path,
            )

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    obj = DataTransformation()
    train_array, test_array, preprocessor_path = obj.initiate_data_transformation(
        os.path.join("artifacts", "train.csv"),
        os.path.join("artifacts", "test.csv"),
    )
    print("✅ Data Transformation complete.")
    print("Preprocessor saved at:", preprocessor_path)
