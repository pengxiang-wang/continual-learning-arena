class UnlearnableEWC(UnlearnableCLAlgorithm, EWC):
    r"""Unlearnable EWC algorithm.

    Variant of EWC that supports unlearning requests and permanent tasks.
    """

    def __init__(
        self,
        backbone: CLBackbone,
        heads: HeadsTIL | HeadsCIL | HeadDIL,
        parameter_change_reg_factor: float,
        when_calculate_fisher_information: str,
        non_algorithmic_hparams: dict[str, Any] = {},
        disable_unlearning: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(
            backbone=backbone,
            heads=heads,
            parameter_change_reg_factor=parameter_change_reg_factor,
            when_calculate_fisher_information=when_calculate_fisher_information,
            non_algorithmic_hparams=non_algorithmic_hparams,
            disable_unlearning=disable_unlearning,
            **kwargs,
        )

        self.valid_task_ids: set[int] = set()
        r"""Task IDs whose Fisher/importances & ref backbones are kept for regularization."""

    def on_train_end(self) -> None:
        super().on_train_end()
        self.valid_task_ids.add(self.task_id)

    def training_step(self, batch: Any) -> dict[str, Tensor]:
        r"""Same as EWC.training_step but regularization sums over valid previous tasks only."""
        x, y = batch

        opt = self.optimizers()
        opt.zero_grad()

        logits, activations = self.forward(x, stage="train", task_id=self.task_id)
        loss_cls = self.criterion(logits, y)

        batch_size = len(y)
        self.num_data += batch_size

        if self.when_calculate_fisher_information == "train":
            loss_cls.backward(retain_graph=True)
            for param_name, param in self.backbone.named_parameters():
                self.parameter_importance[self.task_id][param_name] += (
                    batch_size * param.grad**2
                )

        # FIX: only regularize towards valid (non-unlearned) previous tasks
        loss_reg = 0.0
        for previous_task_id in sorted(self.valid_task_ids):
            if previous_task_id >= self.task_id:
                continue
            if previous_task_id not in self.previous_task_backbones:
                continue
            if previous_task_id not in self.parameter_importance:
                continue

            loss_reg += 0.5 * self.parameter_change_reg(
                target_model=self.backbone,
                ref_model=self.previous_task_backbones[previous_task_id],
                weights=self.parameter_importance[previous_task_id],
            )

        loss = loss_cls + loss_reg

        self.manual_backward(loss)
        opt.step()

        acc = (logits.argmax(dim=1) == y).float().mean()

        return {
            "loss": loss,
            "loss_cls": loss_cls,
            "loss_reg": loss_reg,
            "acc": acc,
            "activations": activations,
        }

    def aggregated_backbone_output(self, input: Tensor) -> Tensor:
        r"""Aggregated backbone output for unlearning metrics.

        EWC keeps a single backbone, so we use its feature directly.
        """
        feature, _ = self.backbone(input, stage="unlearning_test", task_id=self.task_id)
        return feature
