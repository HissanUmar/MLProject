import os
from cnnClassifier.constants import *
from cnnClassifier.utils.common import read_yaml, create_directories
from cnnClassifier.entity.config_entity import DataIngestionConfig, ModelTrainerConfig


class ConfigurationManager:
    def __init__(
        self,
        config_filepath = CONFIG_FILE_PATH,
        params_filepath = PARAMS_FILE_PATH):

        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)

        create_directories([self.config.artifacts_root])


    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion
        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir= config.root_dir,
            local_data_file= config.local_data_file,
            unzip_dir=config.unzip_dir, 
            username = config.username,
            key = config.key,
        )

        return data_ingestion_config


    def get_model_trainer_config(self)-> ModelTrainerConfig:
        config = self.config.model_trainer
        params = self.params.model_trainer

        create_directories([config.root_dir])

        model_trainer_config = ModelTrainerConfig(
            root_dir= config.root_dir,
            train_data_path= config.train_data_path,
            test_data_path= config.test_data_path,
            model_path= config.model_path,
            test_size = params.test_size,
            batch_size = params.batch_size,
            epochs = params.epochs
        )

        return model_trainer_config