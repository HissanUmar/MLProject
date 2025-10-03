from cnnClassifier import logger
from cnnClassifier.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from cnnClassifier.pipeline.stage_02_model_training import ModelTrainingPipeline


# STAGE_NAME = "Data Ingestion stage"

# try:
#     logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
#     obj = DataIngestionTrainingPipeline()
#     obj.main()
#     logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
# except Exception as e:
#     logger.exception(e)
#     raise e


STAGE_NAME = "Model Training stage"

try:
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    obj = ModelTrainingPipeline()
    obj.main()
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise e