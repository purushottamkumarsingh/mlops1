import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from src.exception import CustomException
from src.logger import logging

class DataIngestion:
    def __init__(self):
        self.train_data_path = os.path.join("artifacts", "train.csv")
        self.test_data_path = os.path.join("artifacts", "test.csv")

    def initiate_data_ingestion(self):
        logging.info("Starting data ingestion process...")
        try:
            # ✅ Update path to your dataset
            dataset_path = os.path.join("notebook", "data", "exams.csv")

            df = pd.read_csv(dataset_path)
            logging.info("Dataset loaded successfully.")

            os.makedirs("artifacts", exist_ok=True)

            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            train_set.to_csv(self.train_data_path, index=False, header=True)
            test_set.to_csv(self.test_data_path, index=False, header=True)

            logging.info("Train and test data saved in artifacts folder.")

            return self.train_data_path, self.test_data_path

        except Exception as e:
            logging.error("Error occurred in Data Ingestion")
            raise CustomException(e, sys)

if __name__ == "__main__":
    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()
    print("✅ Data Ingestion Completed")
    print("Train path:", train_data)
    print("Test path:", test_data)
