import os
import zipfile
import gdown
import shutil
from pathlib import Path
from cnnClassifier import logger
from cnnClassifier.utils.common import get_size
from cnnClassifier.entity.config_entity import DataIngestionConfig
from kaggle.api.kaggle_api_extended import KaggleApi



class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config


    def download_file(self)->str:
        '''
        Fetch data from the url
        '''


        try:

            # Auto-detect Git repo root
            PROJECT_ROOT = Path(__file__).resolve()
            for parent in PROJECT_ROOT.parents:
                if (parent / ".git").exists():
                    PROJECT_ROOT = parent
                    break

            print(PROJECT_ROOT)

            home = str(PROJECT_ROOT)
            kaggle_dir = os.path.join(home, ".kaggle")
            kaggle_json_src = os.path.join(home, "kaggle.json")
            kaggle_json_dst = os.path.join(kaggle_dir, "kaggle.json")

            os.makedirs(kaggle_dir, exist_ok=True)

            if os.path.exists(kaggle_json_src):
                shutil.move(kaggle_json_src, kaggle_json_dst)
                print(f"Moved kaggle.json to {kaggle_json_dst}")
            else:
                print("kaggle.json not found in folder!")

            os.chmod(kaggle_json_dst, 0o600)
            print("Permissions set to 600")

            api = KaggleApi()
            api.authenticate()

            logger.info(f"Downloading data into file {self.config.local_data_file}")
            os.makedirs(self.config.root_dir, exist_ok=True)
            api.dataset_download_files(
                "msambare/fer2013",
                path=self.config.local_data_file,
                unzip=False
            )
            logger.info(f"Downloaded data into file {self.config.local_data_file}")

        except Exception as e:
            raise e

    def extract_zip_file(self):
        """
        zip_file_path: str
        Extracts the zip file into the data directory
        Function returns None
        """

        unzip_path = self.config.unzip_dir
        os.makedirs(unzip_path, exist_ok=True)
        with zipfile.ZipFile(os.path.join(self.config.local_data_file, 'fer2013.zip'), 'r') as zip_ref:
            zip_ref.extractall(unzip_path)
