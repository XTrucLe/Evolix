import os

from evolix.config.config import Config
from evolix.engine.trainer import Trainer
from evolix.data.dataset import DataManager
from evolix.utils.checkpoint import CheckpointManager


def trainer():
    try:
        config = Config()
        config.hf_token = os.environ.get("HF_TOKEN", None) if not config.hf_token else config.hf_token
        if not config.hf_token:
            raise ValueError("Hugging Face token is not set. Please set the 'hf_token' in the configuration.")

        data_manager = DataManager(config)
        checkpoint_manager = CheckpointManager(config)
        trainer = Trainer(config, data_manager, checkpoint_manager)
        trainer.run()
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    trainer()
