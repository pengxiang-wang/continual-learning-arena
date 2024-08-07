from typing import Any

from lightning.pytorch.loggers import TensorBoardLogger
import matplotlib.pyplot as plt
import pyrootutils
import torch
from torch import nn

pyrootutils.setup_root(__file__, indicator="pyproject.toml", pythonpath=True)

from src.models import Finetuning
from src.models.calibrators import maskclipper
from src.models.memories import MaskMemory
from src.utils import get_logger

logger = get_logger()


DEFAULT_SMAX = 400.0


class HAT(Finetuning):
    r"""LightningModule for HAT (Hard Attention to Task) continual learning algorithm."""

    def __init__(
        self,
        heads: torch.nn.Module,
        backbone: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        mask_sparsity_reg: torch.nn.Module,
        te_init: str = "N01",
        s_max: float = DEFAULT_SMAX,
        calculate_capacity: bool = False,
        log_capacity: bool = False,
        log_train_mask=False,
    ):
        super().__init__(heads, backbone, optimizer, scheduler)

        # HAT regularisation loss function
        self.mask_sparsity_reg = mask_sparsity_reg

        # Initialisation option for task embedding
        self.te_init = te_init

        # Memory store mask of each task
        self.mask_memory = MaskMemory(
            s_max=self.hparams.s_max, backbone=backbone, approach="hat"
        )

        # manual optimization
        self.automatic_optimization = False

        self.calculate_capacity = calculate_capacity
        self.log_capacity = log_capacity
        self.log_train_mask = log_train_mask

    def forward(self, x: torch.Tensor, task_id: int, scalar: float, stage: str):
        # the forward process propagates input to logits of classes of task_id
        feature, mask = self.backbone(x, scalar, stage)
        logits = self.heads(feature, task_id)
        return logits, mask

    def on_train_start(self):
        for embedding in self.backbone.te.values():
            if self.te_init == "N01":
                nn.init.normal_(embedding.weight, 0, 1)
            elif self.te_init == "U01":
                nn.init.uniform_(embedding.weight, 0, 1)
            elif self.te_init == "U-10":
                nn.init.uniform_(embedding.weight, -1, 0)
            elif self.te_init == "masked":
                pass
            elif self.te_init == "unmasked":
                embedding.weight.data.negative_()

    def on_train_end(self):
        self.mask_memory.update(task_id=self.task_id, backbone=self.backbone)

    def annealed_scalar(self, s_max, batch_idx, num_batches):
        s = 1 / s_max + (s_max - 1 / s_max) * (batch_idx - 1) / (num_batches - 1)
        return s

    def training_step(self, batch: Any, batch_idx: int):
        x, y = batch
        opt = self.optimizers()
        opt.zero_grad()

        num_batches = self.trainer.num_training_batches
        s = self.annealed_scalar(self.hparams.s_max, batch_idx, num_batches)

        # forward step training
        logits, mask = self.forward(x, self.task_id, scalar=s, stage="fit")
        loss_cls = self.criterion(logits, y)

        previous_mask = self.mask_memory.get_union_mask()
        loss_reg, _, _ = self.reg(mask, previous_mask)

        loss_total = loss_cls + loss_reg
        preds = torch.argmax(logits, dim=1)
        targets = y

        # backward step
        self.manual_backward(loss_total)
        capacity = maskclipper.hard_clip_te_masked_gradients(
            self.backbone,
            self.mask_memory,
            self.log_capacity,
        )
        maskclipper.compensate_te_gradients(
            self.backbone, compensate_thres=50, scalar=s, s_max=self.hparams.s_max
        )

        # torch.nn.utils.clip_grad_value_(self.parameters(),self.gradient_clip_val)
        opt.step()

        # log mask
        if self.log_train_mask:
            logger.log_train_mask(
                mask, self.task_id, self.global_step, plot_figure=True
            )

        # log capacity
        if self.log_capacity:
            logger.log_capacity(capacity, self.task_id, self.global_step)

        self.training_step_follow_up(loss_cls, loss_reg, loss_total, preds, targets)

        # return loss or backpropagation will fail
        return loss_total

    def validation_step(self, batch: Any, batch_idx: int):
        x, y = batch

        s = self.hparams.s_max

        # forward step validation
        logits, mask = self.forward(x, self.task_id, scalar=s, stage="fit")
        loss_cls = self.criterion(logits, y)

        previous_mask = self.mask_memory.get_union_mask()
        loss_reg, _, _ = self.reg(mask, previous_mask)

        loss_total = loss_cls + loss_reg
        preds = torch.argmax(logits, dim=1)
        targets = y

        self.validation_step_follow_up(loss_cls, loss_reg, loss_total, preds, targets)

    def on_test_start(self):
        # log test mask
        mask = self.mask_memory.get_mask(self.task_id)
        previous_mask = self.mask_memory.get_union_mask()
        logger.log_test_mask(mask, previous_mask, self.task_id)

    def test_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0):
        x, y = batch

        s = self.hparams.s_max

        test_mask = self.mask_memory.get_mask(dataloader_idx)
        self.backbone.set_test_mask(test_mask)

        logits, _ = self.forward(x, dataloader_idx, scalar=s, stage="test")
        loss_cls = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        targets = y

        self.test_step_follow_up(loss_cls, preds, targets, dataloader_idx, batch)


if __name__ == "__main__":
    _ = HAT(None, None, None, None, None, None)
