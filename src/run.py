"""Training entrypoint for the ETA alignment pipeline.

This module wires together:
1. argument parsing,
2. dataset/resource loading,
3. model/trainer/evaluator initialization,
4. the main training and evaluation loop,
5. runtime state serialization.

The class and method order is organized to match the runtime flow so the file
can be read from top to bottom.
"""

import gc
import json
import math
import os
import time
from datetime import datetime

import torch
from torch.optim.lr_scheduler import (
    ExponentialLR,
    LambdaLR,
)

try:
    from .config import ConfigParser
    from .dataset.loader import load_dataset_state
    from .model import AlignmentModel
    from .runner import Evaluator, Trainer
    from .utils.logging import setup_training_logger, shutdown_logger
    from .utils.metrics import seed_torch
except ImportError:
    from config import ConfigParser
    from dataset.loader import load_dataset_state
    from model import AlignmentModel
    from runner import Evaluator, Trainer
    from utils.logging import setup_training_logger, shutdown_logger
    from utils.metrics import seed_torch


class RunSession:
    """Own the end-to-end lifecycle of one training or evaluation run."""

    def __init__(self):
        """Parse config, prepare logging, and initialize runtime placeholders."""
        super().__init__()
        self.logger = None
        self.args = ConfigParser().parse_args()
        self.args.device = self.resolve_device(self.args.device)
        seed_torch(self.args.seed)
        self.repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        self.start_time = datetime.now()

        if not os.path.isabs(self.args.data_root):
            self.args.data_root = os.path.abspath(
                os.path.join(self.repo_root, self.args.data_root)
            )
        for cache_argument in (
            "dataset_cache_dir",
            "embedding_cache_dir",
            "frozen_evidence_cache",
        ):
            cache_path = getattr(self.args, cache_argument, None)
            if cache_path and not os.path.isabs(cache_path):
                setattr(
                    self.args,
                    cache_argument,
                    os.path.abspath(os.path.join(self.repo_root, cache_path)),
                )
        if self.args.attribute_pretrained_model and not os.path.isabs(
            self.args.attribute_pretrained_model
        ):
            self.args.attribute_pretrained_model = os.path.abspath(
                os.path.join(self.repo_root, self.args.attribute_pretrained_model)
            )
        if self.args.graph_negatives_dir and not os.path.isabs(
            self.args.graph_negatives_dir
        ):
            self.args.graph_negatives_dir = os.path.abspath(
                os.path.join(self.repo_root, self.args.graph_negatives_dir)
            )

        self.logger, self.log_dir, self.log_file_path, self.config_file_path = (
            setup_training_logger(
                self.args,
                self.repo_root,
                self.start_time,
            )
        )

        try:
            self.init_runtime_state()
            self.logger.info(
                "Run initialized at %s", self.start_time.strftime("%Y-%m-%d %H:%M:%S")
            )
            self.logger.info("Logging directory: %s", self.log_dir)
            self.logger.info("Training log file: %s", self.log_file_path)
            self.logger.info("Configuration snapshot: %s", self.config_file_path)
        except Exception:
            shutdown_logger(self.logger)
            raise

    @staticmethod
    def resolve_device(requested_device):
        """Resolve portable device aliases and reject unavailable CUDA devices."""
        normalized = str(requested_device).strip().lower()
        if normalized == "auto":
            return "cuda:0" if torch.cuda.is_available() else "cpu"
        if normalized == "cuda":
            normalized = "cuda:0"
        if normalized.startswith("cuda"):
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "A CUDA device was requested, but CUDA is unavailable. "
                    "Use --device cpu or --device auto."
                )
            try:
                device_index = torch.device(normalized).index
            except (RuntimeError, ValueError) as error:
                raise ValueError(
                    "Invalid --device value: {!r}".format(requested_device)
                ) from error
            device_index = 0 if device_index is None else device_index
            if device_index >= torch.cuda.device_count():
                raise RuntimeError(
                    "CUDA device {} was requested, but only {} device(s) are visible.".format(
                        device_index,
                        torch.cuda.device_count(),
                    )
                )
            return "cuda:{}".format(device_index)
        if normalized != "cpu":
            raise ValueError(
                "Invalid --device value: {!r}; use auto, cpu, cuda, or cuda:N.".format(
                    requested_device
                )
            )
        return normalized

    def __getattr__(self, name):
        """Fallback to parsed arguments to keep call sites concise."""
        if "args" in self.__dict__ and hasattr(self.args, name):
            return getattr(self.args, name)
        raise AttributeError(
            "{} has no attribute '{}'".format(self.__class__.__name__, name)
        )

    def init_runtime_state(self):
        """Initialize mutable runtime members before the run starts."""
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.trainer = None
        self.evaluator = None

        self.dataset_state = None
        self.threshold = self.args.threshold
        self.threshold_weight = self.args.threshold_weight
        self.best_hits_at_1 = -float("inf")
        self.evaluations_without_improvement = 0

        if self.early_stop_patience < 0:
            raise ValueError("--early-stop-patience must be non-negative.")
        if self.load_decision_fusion_model and self.decision_fusion_initial_weights is not None:
            raise ValueError(
                "Use either --load-decision-fusion-model or "
                "--decision-fusion-initial-weights, not both."
            )
        decision_fusion_eval_interval = getattr(self.args, "decision_fusion_eval_interval", 5)
        normalization_temperature = getattr(
            self.args,
            "eval_normalization_temperature",
            0.01,
        )
        normalization_rounds = getattr(
            self.args,
            "eval_normalization_rounds",
            30,
        )
        if decision_fusion_eval_interval <= 0:
            raise ValueError("--decision-fusion-eval-interval must be positive.")
        if normalization_temperature <= 0:
            raise ValueError("--eval-normalization-temperature must be positive.")
        if normalization_rounds <= 0:
            raise ValueError("--eval-normalization-rounds must be positive.")

    def load_dataset(self):
        """Load dataset tensors, mappings, and pseudo-label resources."""
        self.logger.info("Loading dataset resources.")
        self.dataset_state = load_dataset_state(self.args)
        if self.dataset_cache_dir:
            self.logger.info(
                "Dataset tensor cache: %s (%s)",
                self.dataset_cache_dir,
                "hit" if getattr(self.args, "_dataset_cache_hit", False) else "created",
            )
        self.dataset_state.log_summary(self.logger)
        if self.dataset_state.test_pairs is not None:
            torch.save({
                'train': self.dataset_state.supervised_train_pairs,
                'validation': self.dataset_state.dev_pairs,
                'test': self.dataset_state.test_pairs,
                'split_identity': self.dataset_state.split_identity,
            }, os.path.join(self.log_dir, 'alignment_split.pt'))
        self.logger.info("Dataset loading completed.")

    def init_model(self):
        """Create the model and the helper objects that depend on it."""
        self.logger.info("Initializing model.")
        self.model = AlignmentModel(self.args, self.dataset_state).to(self.device)
        if self.train_decision_fusion_only:
            if self.eval_only:
                raise ValueError("--train-decision-fusion-only cannot be combined with --eval-only true.")
            self.model.enable_decision_fusion_only_training()
        self.model.log_summary(self.logger)

        trainable_parameters = [
            parameter
            for parameter in self.model.parameters()
            if parameter.requires_grad
        ]
        trainable_parameter_count = sum(
            parameter.numel() for parameter in trainable_parameters
        )
        self.logger.info("Trainable parameter count: %s", trainable_parameter_count)

        if (
            not self.should_run_evaluation_only()
            and trainable_parameters
        ):
            self.optimizer = torch.optim.AdamW(
                trainable_parameters,
                lr=self.lr,
                weight_decay=self.weight_decay,
            )
            self.scheduler = self.build_scheduler()
        else:
            self.optimizer = None
            self.scheduler = None

        if self.optimizer is not None and self.scheduler is not None:
            self.trainer = Trainer(
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                device=self.device,
                logger=self.logger,
            )
        else:
            self.trainer = None

        self.evaluator = Evaluator(
            model=self.model,
            logger=self.logger,
            export_dev_predictions_path=self.export_dev_predictions,
            export_graph_negatives=self.export_graph_negatives,
            graph_negatives_dir=self.graph_negatives_dir,
            normalization_temperature=self.eval_normalization_temperature,
            normalization_rounds=self.eval_normalization_rounds,
        )

    def build_scheduler(self):
        """Build the configured per-update learning-rate scheduler."""
        if self.scheduler_type == "warmup_cosine":
            return self.build_warmup_cosine_scheduler()
        self.logger.info(
            (
                "Optimizer setup: AdamW(lr=%s, weight_decay=%s), "
                "scheduler=exponential(gamma=%s)"
            ),
            self.lr,
            self.weight_decay,
            self.lr_gamma,
        )
        return ExponentialLR(self.optimizer, gamma=self.lr_gamma)

    def build_warmup_cosine_scheduler(self):
        if not 0.0 <= self.warmup_ratio < 1.0:
            raise ValueError("--warmup-ratio must be in [0, 1).")
        if not 0.0 <= self.min_lr_ratio <= 1.0:
            raise ValueError("--min-lr-ratio must be in [0, 1].")

        steps_per_epoch = max(
            1,
            self.dataset_state.train_batch_count(
                batch_size=self.batch_size,
                balanced=self.balanced_batches,
            ),
        )
        total_steps = max(1, steps_per_epoch * self.epochs)
        warmup_steps = min(
            total_steps - 1,
            max(1, int(math.floor(total_steps * self.warmup_ratio))),
        )
        cosine_steps = max(1, total_steps - warmup_steps)
        min_lr = self.lr * self.min_lr_ratio
        self.logger.info(
            (
                "Optimizer setup: AdamW(lr=%s, weight_decay=%s), "
                "scheduler=warmup_cosine(total_steps=%s, warmup_steps=%s, min_lr=%s)"
            ),
            self.lr,
            self.weight_decay,
            total_steps,
            warmup_steps,
            min_lr,
        )

        def lr_multiplier(current_step):
            if total_steps <= 1:
                return 1.0
            if current_step < warmup_steps:
                return max(1e-3, current_step / float(warmup_steps))
            progress = min(
                1.0,
                (current_step - warmup_steps) / float(cosine_steps),
            )
            cosine_factor = 0.5 * (1.0 + math.cos(math.pi * progress))
            return self.min_lr_ratio + (1.0 - self.min_lr_ratio) * cosine_factor

        return LambdaLR(self.optimizer, lr_lambda=lr_multiplier)

    def active_eval_interval(self):
        if self.train_decision_fusion_only:
            return self.decision_fusion_eval_interval
        return self.eval_interval

    def should_evaluate(self, epoch):
        """Return whether the current epoch should trigger evaluation."""
        interval = self.active_eval_interval()
        return (
            epoch >= self.eval_start_epoch
            and (
                epoch % interval == interval - 1
                or epoch == self.epochs - 1
            )
        )

    def should_run_pseudo_labeling(self, epoch):
        """Return whether pseudo-label generation is enabled for this epoch."""
        return (
            self.should_evaluate(epoch)
            and self.semi_prediction
            and epoch >= self.semi_start_epoch
        )

    def should_run_evaluation_only(self):
        """Return whether this run should load checkpoints and only evaluate."""
        if self.train_decision_fusion_only:
            return False
        return self.eval_only

    def update_best_hits(self, hits_at_1):
        """Track the best dev `hits@1` seen so far."""
        if hits_at_1 > self.best_hits_at_1:
            self.best_hits_at_1 = hits_at_1
            self.evaluations_without_improvement = 0
            return True
        self.evaluations_without_improvement += 1
        return False

    def should_stop_early(self):
        """Return whether validation Hits@1 has exhausted its patience."""
        return (
            self.early_stop_patience > 0
            and self.evaluations_without_improvement
            >= self.early_stop_patience
        )

    def log_graph_layer_weights(self):
        """Log adaptive propagation-layer probabilities for the graph modality."""
        layer_weights = self.model.get_graph_layer_weight_summary()
        if not layer_weights:
            return
        self.logger.info(
            "Graph layer weights: entity=%s relation=%s combined=%s",
            ["{:.4f}".format(value) for value in layer_weights["entity"]],
            ["{:.4f}".format(value) for value in layer_weights["relation"]],
            ["{:.4f}".format(value) for value in layer_weights["combined"]],
        )

    def log_decision_fusion_weights(self):
        """Log aggregate pair-specific weights from Dynamic Decision Fusion."""
        weight_summary = self.model.get_decision_fusion_weight_summary()
        if not weight_summary:
            return
        formatted = {
            module_name: "mean={:.4f}, std={:.4f}, count={}".format(
                values["mean"],
                values["std"],
                values["count"],
            )
            for module_name, values in weight_summary.items()
        }
        self.logger.info("Dynamic Decision Fusion weights: %s", formatted)

    def save_best_models(self):
        """Persist the checkpoints corresponding to the best dev Hits@1 so far."""
        if self.dataset_state.test_pairs is not None and self.final_test:
            torch.save(self.model.state_dict(), os.path.join(self.log_dir, 'best_validation_model.pt'))
        if self.model.has_graph() and not self.train_decision_fusion_only:
            graph_path = os.path.join(self.log_dir, "best_graph_model.pt")
            self.model.save_graph_model(graph_path)
            self.logger.info("Saved best graph modality checkpoint to %s", graph_path)

        if self.model.has_attribute_projector() and not self.train_decision_fusion_only:
            attribute_path = os.path.join(self.log_dir, "best_attribute_model.pt")
            self.model.save_attribute_model(attribute_path)
            self.logger.info(
                "Saved best attribute projector checkpoint to %s",
                attribute_path,
            )

        if self.model.has_visual_projector() and not self.train_decision_fusion_only:
            visual_path = os.path.join(self.log_dir, "best_visual_model.pt")
            self.model.save_visual_model(visual_path)
            self.logger.info("Saved best visual projector checkpoint to %s", visual_path)

        if self.model.has_decision_fusion():
            decision_fusion_path = os.path.join(self.log_dir, "best_dynamic_decision_fusion_model.pt")
            self.model.save_decision_fusion_model(decision_fusion_path)
            self.logger.info("Saved best Dynamic Decision Fusion checkpoint to %s", decision_fusion_path)

    def load_evaluation_checkpoints(self):
        """Load optional initialization/evaluation checkpoints."""
        if self.load_graph_model:
            self.model.load_graph_model(
                self.load_graph_model,
                map_location=self.device,
            )
            self.logger.info(
                "Loaded graph modality checkpoint from %s",
                self.load_graph_model,
            )
        if self.load_attribute_model:
            self.model.load_attribute_model(
                self.load_attribute_model,
                map_location=self.device,
            )
            self.logger.info(
                "Loaded attribute projector checkpoint from %s",
                self.load_attribute_model,
            )
        elif self.attribute_pretrained_model and self.model.has_attribute_projector():
            self.model.load_attribute_model(
                self.attribute_pretrained_model,
                map_location=self.device,
            )
            self.logger.info(
                "Initialized the paper-primary attribute projector from %s",
                self.attribute_pretrained_model,
            )
        if self.load_visual_model:
            self.model.load_visual_model(
                self.load_visual_model,
                map_location=self.device,
            )
            self.logger.info(
                "Loaded visual projector checkpoint from %s",
                self.load_visual_model,
            )
        if self.load_decision_fusion_model:
            self.model.load_decision_fusion_model(
                self.load_decision_fusion_model,
                map_location=self.device,
            )
            self.logger.info("Loaded Dynamic Decision Fusion checkpoint from %s", self.load_decision_fusion_model)
        if self.decision_fusion_anchor_weight > 0:
            anchor_parameter_count = self.model.set_decision_fusion_anchor()
            self.logger.info(
                "Initialized Dynamic Decision Fusion anchor from the loaded run state: "
                "weight=%s parameter_tensors=%s",
                self.decision_fusion_anchor_weight,
                anchor_parameter_count,
            )

    def prepare_frozen_decision_evidence(self):
        """Populate reusable entity evidence for decision-fusion-only training."""
        cache_path = self.frozen_evidence_cache
        if cache_path and os.path.isfile(cache_path):
            try:
                payload = torch.load(
                    cache_path,
                    map_location="cpu",
                    weights_only=False,
                )
            except TypeError:
                payload = torch.load(cache_path, map_location="cpu")
            self.model.load_frozen_decision_evidence_state(payload["state"])
            self.logger.info("Loaded frozen fusion evidence cache: %s", cache_path)
            return
        if not self.train_decision_fusion_only:
            return
        started_at = time.perf_counter()
        cache_sizes = self.model.prepare_frozen_decision_evidence(
            self.dataset_state
        )
        formatted_sizes = {
            module_name: "{:.2f} MiB".format(byte_count / (1024**2))
            for module_name, byte_count in cache_sizes.items()
        }
        self.logger.info(
            "Prepared frozen fusion evidence caches: %s | time=%.4f seconds",
            formatted_sizes,
            time.perf_counter() - started_at,
        )
        if cache_path:
            cache_dir = os.path.dirname(cache_path)
            if cache_dir:
                os.makedirs(cache_dir, exist_ok=True)
            temporary_path = cache_path + ".tmp.{}".format(os.getpid())
            torch.save(
                {
                    "version": 1,
                    "dataset": self.dataset_state.dataset_name,
                    "state": self.model.frozen_decision_evidence_state(),
                },
                temporary_path,
            )
            os.replace(temporary_path, cache_path)
            self.logger.info("Saved frozen fusion evidence cache: %s", cache_path)

    def save_trained_models(self):
        """Persist trained graph, attribute, visual, and decision-fusion components."""
        if self.model.has_graph() and not self.train_decision_fusion_only:
            graph_path = os.path.join(self.log_dir, "graph_model.pt")
            self.model.save_graph_model(graph_path)
            self.logger.info("Saved graph modality checkpoint to %s", graph_path)

        if self.model.has_attribute_projector() and not self.train_decision_fusion_only:
            attribute_path = os.path.join(self.log_dir, "attribute_model.pt")
            self.model.save_attribute_model(attribute_path)
            self.logger.info(
                "Saved attribute projector checkpoint to %s",
                attribute_path,
            )

        if self.model.has_visual_projector() and not self.train_decision_fusion_only:
            visual_path = os.path.join(self.log_dir, "visual_model.pt")
            self.model.save_visual_model(visual_path)
            self.logger.info("Saved visual projector checkpoint to %s", visual_path)

        if self.model.has_decision_fusion():
            decision_fusion_path = os.path.join(self.log_dir, "dynamic_decision_fusion_model.pt")
            self.model.save_decision_fusion_model(decision_fusion_path)
            self.logger.info("Saved Dynamic Decision Fusion checkpoint to %s", decision_fusion_path)

    def run_evaluation_only(self):
        """Evaluate once when the active modalities do not require training."""
        self.logger.info("Running evaluation only.")
        self.model.eval()
        graph_embeddings = None
        if self.export_graph_negatives:
            graph_embeddings = self.evaluator.encode_graph_features(
                dataset_state=self.dataset_state,
            )
        result = self.evaluator.evaluate_dev(
            dataset_state=self.dataset_state,
            graph_embeddings=graph_embeddings,
            return_details=bool(self.export_dev_predictions),
        )
        self.evaluator.export_graph_negatives_if_enabled(
            dataset_state=self.dataset_state,
            graph_embeddings=graph_embeddings,
            epoch=None,
        )
        self.evaluator.log_evaluation_metrics(result, prefix="Evaluation metrics")
        self.log_decision_fusion_weights()
        if self.export_dev_predictions:
            self.evaluator.export_dev_predictions(
                dataset_state=self.dataset_state,
                result=result,
                output_path=self.export_dev_predictions,
            )
        self.update_best_hits(float(result["hits@1"]))
        self.logger.info("Current best Hits@1: %.4f", self.best_hits_at_1)

    def run_final_test(self, restore_best=True):
        """Use held-out labels once, after all training/model selection is done."""
        if self.dataset_state.test_pairs is None or not self.final_test:
            return
        if restore_best:
            path = os.path.join(self.log_dir, 'best_validation_model.pt')
            if not os.path.isfile(path):
                raise RuntimeError("No validation checkpoint was saved; ensure epochs reach eval-start-epoch.")
            self.model.load_state_dict(torch.load(path, map_location=self.device, weights_only=True))
            # Frozen evidence may reflect the last epoch rather than the best one.
            for module in [self.model.graph_modality, *self.model.alignment_modalities.values()]:
                if module is not None:
                    clear = getattr(module, 'clear_frozen_similarity_cache', None)
                    if clear:
                        clear()
        self.model.eval()
        original_pairs = self.dataset_state.dev_pairs
        saved_ids = []
        try:
            self.dataset_state.dev_pairs = self.dataset_state.test_pairs
            for modality in self.model.alignment_modalities.values():
                if getattr(modality, 'uses_precomputed_score', lambda: False)():
                    raise ValueError("Held-out testing requires entity features, not a validation score matrix.")
                saved_ids.append((modality, modality.eval_source_entity_ids, modality.eval_target_entity_ids))
                modality.eval_source_entity_ids = self.dataset_state.test_pairs[:, 0].to(self.device)
                modality.eval_target_entity_ids = self.dataset_state.test_pairs[:, 1].to(self.device)
            result = self.evaluator.evaluate_dev(self.dataset_state)
            self.evaluator.log_evaluation_metrics(result, prefix="Final held-out test metrics")
            with open(os.path.join(self.log_dir, 'test_metrics.json'), 'w') as handle:
                json.dump({'split_identity': self.dataset_state.split_identity,
                           'test_pair_count': len(self.dataset_state.test_pairs),
                           'selection': 'validation_best' if restore_best else 'evaluation_only',
                           'metrics': {k: float(v) for k, v in result.items()}}, handle, indent=2)
        finally:
            self.dataset_state.dev_pairs = original_pairs
            for modality, source_ids, target_ids in saved_ids:
                modality.eval_source_entity_ids = source_ids
                modality.eval_target_entity_ids = target_ids

    def run(self):
        """Execute dataset loading, model setup, training, and evaluation."""
        self.load_dataset()
        self.init_model()
        self.load_evaluation_checkpoints()
        self.prepare_frozen_decision_evidence()

        if self.should_run_evaluation_only() or not self.model.should_run_training_loop():
            self.run_evaluation_only()
            self.run_final_test(restore_best=False)
            return

        if self.trainer is None:
            raise RuntimeError(
                "Trainer is unavailable while the active modality set requires training."
            )

        graph_embeddings = None

        self.logger.info("Starting training loop.")
        if self.early_stop_patience > 0:
            self.logger.info(
                (
                    "Early stopping enabled: patience=%s validation "
                    "evaluations, metric=Hits@1"
                ),
                self.early_stop_patience,
            )
        for epoch in range(self.epochs):
            stop_training = False
            self.trainer.run_epoch(
                dataset_state=self.dataset_state,
                train_loader=self.dataset_state.build_train_loader(
                    batch_size=self.batch_size,
                    shuffle=self.shuffle,
                    balanced=self.balanced_batches,
                ),
                epoch=epoch,
                total_epochs=self.epochs,
            )

            if self.should_evaluate(epoch):
                graph_embeddings, result = self.evaluator.evaluate_epoch(
                    dataset_state=self.dataset_state,
                    epoch=epoch,
                    total_epochs=self.epochs,
                    require_graph_embeddings=(
                        self.should_run_pseudo_labeling(epoch)
                        or self.export_graph_negatives
                    ),
                )
                hits0 = float(result["hits@1"])
                improved = self.update_best_hits(hits0)
                self.log_graph_layer_weights()
                self.log_decision_fusion_weights()
                if improved:
                    self.save_best_models()
                self.logger.info("Updated best Hits@1: %.4f", self.best_hits_at_1)
                stop_training = self.should_stop_early()
                if stop_training:
                    self.logger.info(
                        (
                            "Early stopping triggered at epoch %s/%s after %s "
                            "consecutive evaluations without Hits@1 improvement; "
                            "best Hits@1=%.4f"
                        ),
                        epoch + 1,
                        self.epochs,
                        self.evaluations_without_improvement,
                        self.best_hits_at_1,
                    )

            if self.should_run_pseudo_labeling(epoch):
                self.args._pseudo_epoch = epoch
                pseudo_label_state = self.dataset_state.apply_pseudo_labels(
                    model=self.model,
                    graph_embeddings=graph_embeddings,
                    threshold=self.threshold,
                    threshold_weight=self.threshold_weight,
                    conflict_policy=self.conflict_policy,
                    logger=self.logger,
                )
                self.threshold = pseudo_label_state["threshold"]

            if stop_training:
                break

        self.save_trained_models()
        self.run_final_test()

if __name__ == "__main__":
    """Seed the runtime and execute one `RunSession` with cleanup hooks."""
    model = None

    try:
        model = RunSession()
        model.run()
    except KeyboardInterrupt:
        if model is not None and model.logger is not None:
            model.logger.warning("Training interrupted by user.")

        gc.collect()
        torch.cuda.empty_cache()

        if model is not None and model.logger is not None:
            model.logger.warning(
                "Allocated GPU memory at interruption: %.2f GB",
                torch.cuda.memory_allocated() / (1024**3),
            )
    except Exception:
        if model is not None and model.logger is not None:
            model.logger.exception("Unhandled exception during training.")
        raise
    finally:
        torch.cuda.empty_cache()
        if model is not None:
            shutdown_logger(model.logger)
