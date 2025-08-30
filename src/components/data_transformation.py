import os
import sys
import numpy as np
import pandas as pd
from dataclasses import dataclass
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join("artifacts", "preprocessor.pkl")
    train_array_path: str = os.path.join("artifacts", "train_array.npy")
    test_array_path: str = os.path.join("artifacts", "test_array.npy")
    target_column: str = "math score"   # 👈 update if different


class DataTransformation:
    def __init__(self):
        self.transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self, df: pd.DataFrame):
        try:
            logging.info("Identifying categorical and numerical columns")
            categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
            numerical_cols = df.select_dtypes(exclude=["object"]).columns.tolist()

            if self.transformation_config.target_column in categorical_cols:
                categorical_cols.remove(self.transformation_config.target_column)
            if self.transformation_config.target_column in numerical_cols:
                numerical_cols.remove(self.transformation_config.target_column)

            logging.info(f"Categorical columns: {categorical_cols}")
            logging.info(f"Numerical columns: {numerical_cols}")

            num_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ])

            cat_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
                ("scaler", StandardScaler(with_mean=False))
            ])

            preprocessor = ColumnTransformer(
                transformers=[
                    ("num", num_pipeline, numerical_cols),
                    ("cat", cat_pipeline, categorical_cols)
                ]
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            logging.info("Reading train and test data")
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            target_column = self.transformation_config.target_column
            if target_column not in train_df.columns:
                raise CustomException(
                    f"Target column '{target_column}' not found in dataset. "
                    f"Available columns: {train_df.columns.tolist()}",
                    sys
                )

            logging.info("Obtaining preprocessing object")
            preprocessing_obj = self.get_data_transformer_object(train_df)

            X_train = train_df.drop(columns=[target_column], axis=1)
            y_train = train_df[target_column]

            X_test = test_df.drop(columns=[target_column], axis=1)
            y_test = test_df[target_column]

            logging.info("Applying preprocessing object on training and test data")
            X_train_arr = preprocessing_obj.fit_transform(X_train)
            X_test_arr = preprocessing_obj.transform(X_test)

            train_arr = np.c_[X_train_arr, np.array(y_train)]
            test_arr = np.c_[X_test_arr, np.array(y_test)]

            # Save arrays
            np.save(self.transformation_config.train_array_path, train_arr)
            np.save(self.transformation_config.test_array_path, test_arr)

            # Save preprocessor
            save_object(
                file_path=self.transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            logging.info("Preprocessor and transformed arrays saved successfully")

            return train_arr, test_arr, self.transformation_config.preprocessor_obj_file_path

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    from src.components.data_ingestion import DataIngestion

    ingestion = DataIngestion()
    train_path, test_path = ingestion.initiate_data_ingestion()

    transformation = DataTransformation()
    train_arr, test_arr, preprocessor_path = transformation.initiate_data_transformation(train_path, test_path)

    print("✅ Data transformation completed")
    print(f"Train array shape: {train_arr.shape}")
    print(f"Test array shape: {test_arr.shape}")
    print(f"Preprocessor saved at: {preprocessor_path}")
