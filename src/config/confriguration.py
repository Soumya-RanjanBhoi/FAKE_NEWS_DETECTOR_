from src.constants import *
from src.utils.common import read_yaml,create_directories
from src.entity import ModelConfiguration

class ConfriguationManager:
    def __init__(self,config_file_path=CONFIG_FILE_PATH):
        self.config=read_yaml(config_file_path)


    def get_model_evaluation_config(self)->ModelConfiguration:
        config=self.config.ModelConfig

        model_config=ModelConfiguration(
            root_dir=config.root_dir,
            tokenizer_path=config.tokenizer_path,
            model_path=config.model_path
        )

        return model_config