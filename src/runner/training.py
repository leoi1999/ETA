import time

import torch


class Trainer:
    def __init__(self, model, optimizer, scheduler, device, logger):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.logger = logger

    def run_epoch(self, dataset_state, train_loader, epoch, total_epochs):
        started_at = time.time()
        self.logger.info("Epoch %s/%s started.", epoch + 1, total_epochs)
        if epoch == 0:
            batch_sizes = getattr(
                getattr(train_loader, "batch_sampler", None),
                "batch_sizes",
                None,
            )
            if batch_sizes:
                self.logger.info(
                    "Balanced training batches: count=%s min_size=%s max_size=%s",
                    len(batch_sizes),
                    min(batch_sizes),
                    max(batch_sizes),
                )
        self.model.train()
        total_loss = 0.0
        step_count = 0
        fusion_grad_norm_total = 0.0

        for pair_batch in train_loader:
            if len(pair_batch) == 0:
                continue

            pair_batch = pair_batch.to(self.device)
            self.optimizer.zero_grad(set_to_none=True)
            loss = self.model.compute_training_loss(
                train_pairs=pair_batch,
                dataset_state=dataset_state,
            )
            loss.backward()
            if getattr(self.model, "decision_fusion_only_training", False):
                squared_gradient_norm = torch.zeros((), device=loss.device)
                for parameter in self.model.decision_fusion.parameters():
                    if parameter.grad is not None:
                        squared_gradient_norm = (
                            squared_gradient_norm + parameter.grad.detach().square().sum()
                        )
                fusion_grad_norm_total += float(torch.sqrt(squared_gradient_norm).cpu())
            self.optimizer.step()
            self.scheduler.step()
            total_loss += float(loss.detach().cpu())
            step_count += 1

        ended_at = time.time()
        if step_count:
            self.logger.info(
                "Epoch %s/%s average training loss: %.6f",
                epoch + 1,
                total_epochs,
                total_loss / step_count,
            )
            if getattr(self.model, "decision_fusion_only_training", False):
                self.logger.info(
                    "Epoch %s/%s average Dynamic Decision Fusion gradient norm: %.6f",
                    epoch + 1,
                    total_epochs,
                    fusion_grad_norm_total / step_count,
                )
        self.logger.info("Epoch %s/%s training time: %.4f seconds", epoch + 1, total_epochs, ended_at - started_at)
