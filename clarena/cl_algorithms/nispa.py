# """
# The submodule in `cl_algorithms` for NISPA (Neuro-Inspired Stability-Plasticity Adaptation) algorithm.
# """

# __all__ = ["NISPA"]

# import logging
# import math
# import random
# from copy import deepcopy
# from typing import Any, Dict

# import torch
# from torch import Tensor

# from clarena.backbones import NISPAMaskBackbone
# from clarena.cl_algorithms import CLAlgorithm
# from clarena.heads import HeadsTIL

# # built-in logger
# pylogger = logging.getLogger(__name__)


# class NISPA(CLAlgorithm):
#     r"""[NISPA (Neuro-Inspired Stability-Plasticity Adaptation)](https://proceedings.mlr.press/v162/gurbuz22a.html) algorithm.

#     An architecture-based approach that selects neurons and weights through manual rules.
#     """

#     def __init__(
#         self,
#         backbone: NISPAMaskBackbone,
#         heads: HeadsTIL,
#         num_epochs_per_phase: int,
#         accuracy_fall_threshold: float,
#         k: float,
#         non_algorithmic_hparams: dict[str, Any] = {},
#     ) -> None:
#         super().__init__(
#             backbone=backbone,
#             heads=heads,
#             non_algorithmic_hparams=non_algorithmic_hparams,
#         )

#         # save additional algorithmic hyperparameters
#         self.save_hyperparameters(
#             "num_epochs_per_phase",
#             "accuracy_fall_threshold",
#             "k",
#         )

#         print(backbone)
#         self.num_epochs_per_phase = num_epochs_per_phase
#         self.accuracy_fall_threshold = accuracy_fall_threshold
#         self.k = k

#         # unit-level masks (size = #neurons in layer)
#         self.candidate_stable_unit_mask_t: Dict[str, Tensor] = {}
#         self.stable_unit_mask_t: Dict[str, Tensor] = {}
#         self.plastic_unit_mask_t: Dict[str, Tensor] = {}

#         # bookkeeping for phases
#         self.best_phase_acc: float
#         self.phase_idx: int

#     def on_train_start(self) -> None:
#         """Initialize all masks at the very beginning of Task 1."""
#         self.best_phase_acc = 0.0
#         self.phase_idx = 0

#         if self.task_id == 1:
#             # zero‐out the backbone’s parameter masks
#             self.backbone.initialize_parameter_mask()

#             for layer in self.backbone.weighted_layer_names:
#                 # number of units = output dim of that layer
#                 num_units = self.backbone.get_layer_by_name(layer).weight.shape[0]
#                 self.candidate_stable_unit_mask_t[layer] = torch.zeros(
#                     num_units, device=self.device
#                 )
#                 self.stable_unit_mask_t[layer] = torch.zeros(
#                     num_units, device=self.device
#                 )
#                 self.plastic_unit_mask_t[layer] = torch.ones(
#                     num_units, device=self.device
#                 )

#     def clip_grad_by_frozen_mask(self) -> None:
#         """Zero‐out grads on frozen connections."""
#         for layer_name in self.backbone.weighted_layer_names:
#             layer = self.backbone.get_layer_by_name(layer_name)
#             layer.weight.grad.data *= (
#                 1.0 - self.backbone.frozen_weight_mask_t[layer_name]
#             )
#             if layer.bias is not None:
#                 layer.bias.grad.data *= (
#                     1.0 - self.backbone.frozen_bias_mask_t[layer_name]
#                 )

#     def forward(
#         self,
#         input: torch.Tensor,
#         stage: str,
#         task_id: int | None = None,
#     ) -> tuple[Tensor, Dict[str, Tensor], Dict[str, Tensor], Dict[str, Tensor]]:
#         """Wrapper around the backbone’s forward + the heads."""
#         features, w_mask, b_mask, activations = self.backbone(input, stage=stage)
#         logits = self.heads(features, task_id)
#         return logits, w_mask, b_mask, activations

#     def training_step(self, batch: Any) -> dict[str, Tensor]:
#         x, y = batch
#         opt = self.optimizers()
#         opt.zero_grad()

#         logits, w_mask, b_mask, _ = self.forward(x, stage="train", task_id=self.task_id)
#         loss = self.criterion(logits, y)
#         self.manual_backward(loss)
#         self.clip_grad_by_frozen_mask()
#         opt.step()

#         acc = (logits.argmax(dim=1) == y).float().mean()
#         return {"loss": loss, "acc": acc}

#     def on_validation_epoch_end(self, outputs: dict[str, Any]) -> None:
#         """Every num_epochs_per_phase epochs, do a phase‐end check, selection, and rewire."""
#         if (self.current_epoch + 1) % self.num_epochs_per_phase != 0:
#             return

#         # cache model + optimizer in case we need to rollback
#         cached = {
#             "model": deepcopy(self.state_dict()),
#             "opt": deepcopy(self.trainer.optimizers[0].state_dict()),
#         }

#         val_acc = outputs["acc"].item()
#         self.best_phase_acc = max(self.best_phase_acc, val_acc)

#         if val_acc >= self.best_phase_acc - self.accuracy_fall_threshold:
#             # still “good enough” → select + rewire
#             tau = 0.5 * (1 + math.cos(self.phase_idx * math.pi / self.k))
#             self.select_candidate_stable_units(activation_fraction=tau)

#             # merge candidate → stable, update plastic = complement
#             for layer in self.backbone.weighted_layer_names:
#                 self.stable_unit_mask_t[layer] = (
#                     self.stable_unit_mask_t[layer]
#                     | self.candidate_stable_unit_mask_t[layer]
#                 ).float()
#                 self.plastic_unit_mask_t[layer] = 1.0 - self.stable_unit_mask_t[layer]

#             # update the backbone’s parameter‐level masks from these unit‐masks
#             self._update_parameter_masks()

#             # drop & grow
#             dropped = self.drop_connections_plastic_to_stable()
#             self.grow_new_connections(dropped)

#             self.phase_idx += 1

#         else:
#             # roll back to previous checkpoint
#             self.load_state_dict(cached["model"])
#             self.trainer.optimizers[0].load_state_dict(cached["opt"])
#             pylogger.info(f"Phase {self.phase_idx} terminated early (val_acc fell).")

#     def _update_parameter_masks(self) -> None:
#         """Convert each layer’s unit‐mask → a weight/bias mask."""
#         for layer in self.backbone.weighted_layer_names:
#             # gather shapes
#             W = self.backbone.weight_mask_t[layer]
#             shape = W.shape
#             stable_units = self.stable_unit_mask_t[layer].bool()
#             plastic_units = self.plastic_unit_mask_t[layer].bool()

#             # out-unit plastic mask
#             out_pl = plastic_units.view(-1, 1).expand(shape)

#             # in-unit plastic mask (preceding layer)
#             prev = self.backbone.preceding_layer_name(layer)
#             if prev:
#                 in_pl = self.plastic_unit_mask_t[prev].view(1, -1).expand(shape)
#             else:
#                 in_pl = torch.ones(shape, device=self.device, dtype=torch.bool)

#             # new weight mask = plastic→plastic
#             self.backbone.weight_mask_t[layer] = (out_pl & in_pl).float()
#             # bias mask = plastic units only
#             self.backbone.bias_mask_t[layer] = plastic_units.float()

#     def select_candidate_stable_units(self, activation_fraction: float) -> None:
#         """Pick the top‐activated units until we reach `activation_fraction` of total."""
#         # accumulate per‐unit activations over the training set
#         summed = {
#             layer: torch.zeros_like(self.stable_unit_mask_t[layer])
#             for layer in self.backbone.weighted_layer_names
#         }
#         total_data = 0

#         for x, _ in self.trainer.datamodule.train_dataloader():
#             x = x.to(self.device)
#             _, _, _, acts = self.forward(x, stage="validation")
#             batch = x.shape[0]
#             total_data += batch

#             for layer in summed:
#                 # sum up the ∥activation∥ per unit
#                 summed[layer] += acts[layer].sum(dim=0)  # sum over batch

#         # threshold per‐layer
#         for layer in summed:
#             vals = summed[layer]
#             cutoff = vals.sum() * activation_fraction
#             cum = 0.0
#             mask = torch.zeros_like(vals)

#             # pick largest values until cum ≥ cutoff
#             while cum < cutoff:
#                 idx = torch.argmax(vals)
#                 cum += vals[idx].item()
#                 mask[idx] = 1.0
#                 vals[idx] = 0.0  # ensure we don’t pick it twice

#             self.candidate_stable_unit_mask_t[layer] = mask.to(self.device)

#     def drop_connections_plastic_to_stable(self) -> Dict[str, int]:
#         """
#         Drop connections *from* plastic units *into* stable units:
#         i.e. weight[stable_out, plastic_in] → 0, bias[stable_out] → 0.
#         """
#         dropped_counts: Dict[str, int] = {}

#         for layer in self.backbone.weighted_layer_names:
#             Wm = self.backbone.weight_mask_t[layer]
#             bm = self.backbone.bias_mask_t[layer]

#             stable_out = self.stable_unit_mask_t[layer].bool()
#             prev = self.backbone.preceding_layer_name(layer)
#             if prev:
#                 plastic_in = (~self.stable_unit_mask_t[prev]).bool()
#             else:
#                 plastic_in = torch.zeros(
#                     Wm.shape[1], device=self.device, dtype=torch.bool
#                 )

#             before_w = int(Wm.sum().item())
#             # zero out those connections
#             Wm[stable_out, :][:, plastic_in] = 0
#             after_w = int(Wm.sum().item())

#             before_b = int(bm.sum().item())
#             bm[stable_out] = 0
#             after_b = int(bm.sum().item())

#             self.backbone.weight_mask_t[layer] = Wm
#             self.backbone.bias_mask_t[layer] = bm
#             dropped_counts[layer] = (before_w - after_w) + (before_b - after_b)

#         return dropped_counts

#     def grow_new_connections(self, dropped: Dict[str, int]) -> None:
#         """
#         Regrow new connections *among plastic units* to keep the network at a fixed density.
#         For simplicity, we randomly reassign the dropped number of connections.
#         """
#         for layer, num_drop in dropped.items():
#             if num_drop <= 0:
#                 continue

#             Wm = self.backbone.weight_mask_t[layer]
#             prev = self.backbone.preceding_layer_name(layer)

#             stable_out = self.stable_unit_mask_t[layer].bool()
#             plastic_out = ~stable_out
#             if prev:
#                 plastic_in = (~self.stable_unit_mask_t[prev]).bool()
#             else:
#                 plastic_in = torch.ones(
#                     Wm.shape[1], device=self.device, dtype=torch.bool
#                 )

#             # find all available plastic→plastic positions that are currently zero
#             mask = (Wm == 0) & plastic_out.view(-1, 1) & plastic_in.view(1, -1)
#             coords = torch.nonzero(mask, as_tuple=False)
#             if coords.numel() == 0:
#                 continue

#             coords = coords.tolist()
#             chosen = random.sample(coords, min(len(coords), num_drop))
#             for i, j in chosen:
#                 Wm[i, j] = 1.0

#             # for bias, simply ensure all plastic units have bias = 1
#             bm = self.backbone.bias_mask_t[layer]
#             bm[plastic_out] = 1.0

#             self.backbone.weight_mask_t[layer] = Wm
#             self.backbone.bias_mask_t[layer] = bm

#     def on_train_end(self) -> None:
#         """
#         Called at the very end of a task:
#         freeze *all* connections that were used (mask = 1) as part of this task.
#         """
#         for layer in self.backbone.weighted_layer_names:
#             # freeze any connection that was ever allowed (weight_mask_t == 1)
#             self.backbone.frozen_weight_mask_t[layer] |= (
#                 self.backbone.weight_mask_t[layer] > 0
#             ).float()
#             self.backbone.frozen_bias_mask_t[layer] |= (
#                 self.backbone.bias_mask_t[layer] > 0
#             ).float()

#     def validation_step(self, batch: Any) -> dict[str, Tensor]:
#         x, y = batch
#         logits, _, _, _ = self.forward(x, stage="validation")
#         loss = self.criterion(logits, y)
#         acc = (logits.argmax(dim=1) == y).float().mean()
#         return {"loss": loss, "acc": acc}

#     def test_step(
#         self, batch: Any, batch_idx: int, dataloader_idx: int = 0
#     ) -> dict[str, Tensor]:
#         x, y = batch
#         logits, _, _, _ = self.forward(x, stage="test")
#         loss = self.criterion(logits, y)
#         acc = (logits.argmax(dim=1) == y).float().mean()
#         return {"loss": loss, "acc": acc}
