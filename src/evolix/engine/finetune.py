from typing import Type
 
from evolix.config.finetune_config import FinetuneConfig
from evolix.engine.trainer import Trainer


class FinetuneTrainer(Trainer):
    def __init__(self, config: Type[FinetuneConfig], data_manager, checkpoint_manager, base_checkpoint_manager=None):
        super().__init__(config, data_manager, checkpoint_manager)
        self.base_checkpoint_manager = base_checkpoint_manager

    def _load_initial_weights(self, model, optimizer, scaler, dataset, ddp_world_size) -> int:
        if self.config.resume:
            return self.checkpoint_manager.load(model, optimizer, scaler, dataset, ddp_world_size) + 1

        if self.base_checkpoint_manager is not None:
            self.base_checkpoint_manager.load(model, optimizer=None, scaler=None)
 
        return 0
