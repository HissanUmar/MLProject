from cnnClassifier import logger
from cnnClassifier.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from cnnClassifier.pipeline.stage_02_model_training import ModelTrainingPipeline
import dagshub
import mlflow, os



os.environ["MLFLOW_TRACKING_USERNAME"] = "hissanumar"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "f4fdd40f194f219eb2b51aaa3c0aaded45e9141c"



dagshub.init(repo_owner="hissanumar", repo_name="MLProject", mlflow=True)
mlflow.set_tracking_uri("https://dagshub.com/hissanumar/MLProject.mlflow")
mlflow.set_experiment("FacialExpressionClassifier")

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