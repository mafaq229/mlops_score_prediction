import sys

from src.exception import CustomException
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer


class TrainPipeline:
    def __init__(self):
        pass
    
    def train(self):
        obj = DataIngestion()
        train_data, test_data = obj.initiate_data_ingestion()
        
        data_transformation = DataTransformation()
        train_arr, test_arr, _ = data_transformation.initiate_data_transformation(train_data, test_data)
        
        model_trainer = ModelTrainer()
        score = model_trainer.initiate_model_trainer(train_arr, test_arr)
        return score
    

if __name__ == "__main__":
    train_pipeline = TrainPipeline()
    train_pipeline.train()
    print("Training Complete")