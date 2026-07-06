import os

from evolix.config.config import Config
from evolix.data.dataset import DataManager
from evolix.engine.evaluator import Evaluator
from evolix.utils.checkpoint import CheckpointManager


def evaluate():
    try:
        cfg = Config()
        cfg.data_split = "validation"
        cfg.hf_token = os.environ.get("HF_TOKEN", None) if not cfg.hf_token else cfg.hf_token
        if not cfg.hf_token:
            raise ValueError("Hugging Face token is not set. Please set the 'hf_token' in the configuration.")

        data_manager = DataManager(cfg)
        checkpoint_manager = CheckpointManager(cfg, require_write=False)

        evaluator = Evaluator(cfg, data_manager, checkpoint_manager)
        evaluator.run()
    except Exception as e:
        print(f"Error: {e}")
        raise


if __name__ == "__main__":
    evaluate()
