import torch
import torch.nn.functional as F

from .decision_fusion import (
    DynamicDecisionFusion,
)
from .graph import BidirectionalGraphAlignmentLoss
from .modalities import (
    GraphModalityModule,
    load_alignment_modality_modules,
    parse_active_modules,
)


class AlignmentModel(torch.nn.Module):
    CHECKPOINT_VERSION = 2

    def __init__(self, args, dataset_state):
        super().__init__()
        self.args = args
        self.dataset_name = dataset_state.dataset_name
        self.active_modules = parse_active_modules(args.modules)
        self.graph_modality = None
        self.alignment_modalities = torch.nn.ModuleDict()
        self.loss_module = None
        self.decision_fusion_loss_weight = args.decision_fusion_loss_weight
        self.decision_fusion_balance_weight = args.decision_fusion_balance_weight
        self.decision_fusion_anchor_weight = args.decision_fusion_anchor_weight
        self.decision_score_temperature = args.decision_score_temperature
        self.decision_fusion_detach_scores = args.decision_fusion_detach_scores
        if self.decision_score_temperature <= 0:
            raise ValueError("decision-score temperature must be positive")
        if (
            self.decision_fusion_loss_weight < 0
            or self.decision_fusion_balance_weight < 0
            or self.decision_fusion_anchor_weight < 0
        ):
            raise ValueError("decision-fusion loss weights must be non-negative")
        self.decision_fusion_training_enabled = False
        self.decision_fusion_only_training = False
        self.decision_fusion = self._build_decision_fusion(args)
        self._decision_fusion_anchor = None
        self.modality_load_time = 0.0

        self._init_modalities(dataset_state)
        self._init_loss_module(dataset_state)
        self._configure_decision_fusion_training()

    @staticmethod
    def _build_decision_fusion(args):
        return DynamicDecisionFusion(
            hidden_dim=args.decision_fusion_hidden_dim,
            temperature=args.decision_fusion_temperature,
            chunk_size=args.decision_fusion_chunk_size,
            initial_weights=args.decision_fusion_initial_weights,
        )

    def _configure_decision_fusion_training(self):
        score_source_count = int(self.has_graph())
        for modality in self.alignment_modalities.values():
            supports_decision_fusion_training = getattr(
                modality,
                "supports_decision_fusion_training",
                lambda: False,
            )
            score_source_count += int(supports_decision_fusion_training())

        self.decision_fusion_training_enabled = (
            self.decision_fusion_loss_weight > 0 and score_source_count >= 2
        )
        if not self.decision_fusion_training_enabled:
            for parameter in self.decision_fusion.parameters():
                parameter.requires_grad_(False)

    def _init_modalities(self, dataset_state):
        if "graph" in self.active_modules:
            self.graph_modality = GraphModalityModule(self.args, dataset_state)

        self.alignment_modalities, self.modality_load_time = load_alignment_modality_modules(
            args=self.args,
            dataset_state=dataset_state,
            active_modules=self.active_modules,
        )

    def _init_loss_module(self, dataset_state):
        if self.has_graph():
            self.loss_module = BidirectionalGraphAlignmentLoss(
                entity_count=dataset_state.entity_count,
                source_entity_ids=dataset_state.source_entity_ids,
                target_entity_ids=dataset_state.target_entity_ids,
                beta=self.args.hard_negative_beta,
            )

    def has_graph(self):
        return self.graph_modality is not None

    def has_decision_fusion(self):
        return self.decision_fusion is not None

    def has_attribute_projector(self):
        """Return whether the active attribute modality owns a learned head."""
        if "attribute" not in self.alignment_modalities:
            return False
        return bool(self.alignment_modalities["attribute"].supports_training())

    def has_visual_projector(self):
        """Return whether the active visual modality owns a learned head."""
        if "visual" not in self.alignment_modalities:
            return False
        return bool(self.alignment_modalities["visual"].supports_training())

    def should_run_training_loop(self):
        """Decide whether the active modality set requires training.

        The orchestration layer should not need to know which concrete
        modalities are trainable. It only asks the model whether a training
        loop makes sense for the currently active configuration.
        """
        return self.has_trainable_parameters()

    def has_trainable_parameters(self):
        return any(parameter.requires_grad for parameter in self.parameters())

    def enable_decision_fusion_only_training(self):
        if not self.decision_fusion_training_enabled:
            raise RuntimeError(
                "Decision-fusion-only training requires at least two safe score sources."
            )

        self.decision_fusion_only_training = True
        for parameter_name, parameter in self.named_parameters():
            parameter.requires_grad_(parameter_name.startswith("decision_fusion."))
        # The model is constructed in train mode. Re-apply the mode so frozen
        # evidence encoders immediately switch off dropout and other updates.
        self.train(True)

    def set_decision_fusion_anchor(self):
        """Snapshot the current dynamic decision fusion parameters for stable continuation."""
        self._decision_fusion_anchor = {
            parameter_name: parameter.detach().clone()
            for parameter_name, parameter in self.decision_fusion.named_parameters()
        }
        return len(self._decision_fusion_anchor)

    def compute_decision_fusion_anchor_loss(self):
        """Penalize movement away from the decision-fusion state loaded at run start."""
        anchor = getattr(self, "_decision_fusion_anchor", None)
        if not anchor:
            raise RuntimeError(
                "Fusion anchor loss is enabled before a decision-fusion anchor was initialized."
            )

        anchor_loss = None
        for parameter_name, parameter in self.decision_fusion.named_parameters():
            reference = anchor.get(parameter_name)
            if reference is None:
                raise RuntimeError(
                    "Dynamic Decision Fusion parameter '{}' is missing from the anchor.".format(
                        parameter_name
                    )
                )
            parameter_loss = torch.mean((parameter - reference) ** 2)
            anchor_loss = (
                parameter_loss
                if anchor_loss is None
                else anchor_loss + parameter_loss
            )
        if anchor_loss is None:
            raise RuntimeError("Cannot anchor a Dynamic Decision Fusion without parameters.")
        return anchor_loss

    @torch.inference_mode()
    def prepare_frozen_decision_evidence(self, dataset_state):
        """Precompute entity representations used by a frozen decision-fusion-only run."""
        if not self.decision_fusion_only_training:
            return {}

        cache_sizes = {}
        if self.graph_modality is not None:
            prepare_cache = getattr(
                self.graph_modality,
                "prepare_frozen_similarity_cache",
                None,
            )
            if prepare_cache is not None:
                cache_sizes["graph"] = int(prepare_cache(dataset_state))

        for module_name, modality in self.alignment_modalities.items():
            prepare_cache = getattr(
                modality,
                "prepare_frozen_similarity_cache",
                None,
            )
            if prepare_cache is not None:
                cache_sizes[module_name] = int(prepare_cache(dataset_state))
        return cache_sizes

    @staticmethod
    def _move_cached_value(value, device):
        if isinstance(value, torch.Tensor):
            return value.detach().to(device)
        if isinstance(value, tuple):
            return tuple(
                AlignmentModel._move_cached_value(item, device) for item in value
            )
        if isinstance(value, list):
            return [
                AlignmentModel._move_cached_value(item, device) for item in value
            ]
        if isinstance(value, dict):
            return {
                key: AlignmentModel._move_cached_value(item, device)
                for key, item in value.items()
            }
        return value

    def frozen_decision_evidence_state(self):
        """Return CPU tensors that can be reused by another frozen run."""
        state = {}
        if self.graph_modality is not None:
            state["graph"] = self._move_cached_value(
                getattr(self.graph_modality, "_frozen_similarity_cache", None),
                "cpu",
            )
        for module_name, modality in self.alignment_modalities.items():
            state[module_name] = self._move_cached_value(
                getattr(modality, "_frozen_entity_cache", None),
                "cpu",
            )
        return state

    def load_frozen_decision_evidence_state(self, state):
        """Restore cached entity evidence on the model's current device."""
        device = next(self.parameters()).device
        if self.graph_modality is not None and "graph" in state:
            self.graph_modality._frozen_similarity_cache = self._move_cached_value(
                state["graph"], device
            )
        for module_name, modality in self.alignment_modalities.items():
            if module_name in state:
                modality._frozen_entity_cache = self._move_cached_value(
                    state[module_name], device
                )

    def train(self, mode=True):
        super().train(mode)
        if mode and self.decision_fusion_only_training:
            if self.graph_modality is not None:
                self.graph_modality.eval()
            self.alignment_modalities.eval()
            self.decision_fusion.train(True)
        return self

    def log_summary(self, logger):
        logger.info("Active model modules: %s", ", ".join(self.active_modules) if self.active_modules else "none")
        logger.info(
            "Decision fusion: dynamic | supervised training: %s | decision-fusion-only: %s",
            self.decision_fusion_training_enabled,
            self.decision_fusion_only_training,
        )
        initial_weights = getattr(self.decision_fusion, "initial_weights", None)
        if initial_weights is not None:
            logger.info(
                "Dynamic Decision Fusion global initialization: %s",
                {
                    module_name: "{:.4f}".format(initial_weights[index])
                    for index, module_name in enumerate(self.decision_fusion.SCORE_ORDER)
                },
            )
        if self.graph_modality is not None:
            logger.info("Initialized graph modality '%s'.", self.graph_modality.modality_name)
        for module_name, modality in self.alignment_modalities.items():
            training_status = "trainable" if modality.supports_training() else "frozen"
            logger.info(
                "Initialized %s modality '%s' (%s).",
                training_status,
                module_name,
                modality.modality_name,
            )
        logger.info("Modality initialization time: %.4f seconds", self.modality_load_time)

    def _checkpoint_payload(self, module_name, module):
        payload = {
            "checkpoint_version": self.CHECKPOINT_VERSION,
            "dataset_name": self.dataset_name,
            "active_modules": list(self.active_modules),
            "module_name": module_name,
            "module_class": module.__class__.__name__,
            "state_dict": module.state_dict(),
        }
        if module_name == "decision_fusion":
            payload["score_order"] = list(self.decision_fusion.SCORE_ORDER)
        elif module_name == "graph":
            payload["modality_name"] = self.graph_modality.modality_name
        return payload

    @staticmethod
    def _extract_state_dict(checkpoint):
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            return checkpoint["state_dict"]
        return checkpoint

    def save_graph_model(self, checkpoint_path):
        if not self.has_graph():
            raise RuntimeError("Cannot save graph checkpoint: graph modality is inactive.")
        torch.save(
            self._checkpoint_payload("graph", self.graph_modality),
            checkpoint_path,
        )
        return checkpoint_path

    def load_graph_model(self, checkpoint_path, map_location=None):
        if not self.has_graph():
            raise RuntimeError(
                "Cannot load graph checkpoint without enabling the graph module."
            )
        checkpoint = torch.load(
            checkpoint_path,
            map_location=map_location,
            weights_only=True,
        )
        state_dict = self._extract_state_dict(checkpoint)
        self.graph_modality.clear_frozen_similarity_cache()
        self.graph_modality.load_state_dict(state_dict)
        return checkpoint

    def _attribute_checkpoint_payload(self):
        if "attribute" not in self.alignment_modalities:
            raise RuntimeError(
                "Cannot save attribute checkpoint: attribute modality is inactive."
            )
        modality = self.alignment_modalities["attribute"]
        if not modality.supports_training():
            raise RuntimeError(
                "Cannot save attribute checkpoint without enabling the attribute projector."
            )

        feature_metadata = modality.attribute_metadata or {}
        return {
            "checkpoint_version": self.CHECKPOINT_VERSION,
            "dataset_name": self.dataset_name,
            "active_modules": list(self.active_modules),
            "module_name": "attribute",
            "module_class": modality.__class__.__name__,
            "modality_name": modality.modality_name,
            # Frozen entity features are intentionally excluded. They are
            # reloaded from attribute_input_path when the model is constructed.
            "state_dict": modality.projector.state_dict(),
            "attribute_input_path": modality.attribute_input_path,
            "input_embedding_dim": int(modality.attribute_features.shape[1]),
            "representation": feature_metadata.get("representation"),
            "pooling": feature_metadata.get("pooling"),
            "text_content": feature_metadata.get("text_content"),
        }

    def save_attribute_model(self, checkpoint_path):
        torch.save(self._attribute_checkpoint_payload(), checkpoint_path)
        return checkpoint_path

    def load_attribute_model(self, checkpoint_path, map_location=None):
        if "attribute" not in self.alignment_modalities:
            raise RuntimeError(
                "Cannot load attribute checkpoint without enabling the attribute module."
            )
        modality = self.alignment_modalities["attribute"]
        if not modality.supports_training():
            raise RuntimeError(
                "Cannot load attribute checkpoint without enabling the attribute projector."
            )

        checkpoint = torch.load(
            checkpoint_path,
            map_location=map_location,
            weights_only=True,
        )
        if isinstance(checkpoint, dict):
            checkpoint_module_name = checkpoint.get("module_name")
            if checkpoint_module_name and checkpoint_module_name != "attribute":
                raise RuntimeError(
                    "Checkpoint module {} is not an attribute checkpoint.".format(
                        checkpoint_module_name
                    )
                )
            checkpoint_class = checkpoint.get("module_class")
            if checkpoint_class and checkpoint_class != modality.__class__.__name__:
                raise RuntimeError(
                    "Attribute checkpoint class {} does not match active module {}.".format(
                        checkpoint_class,
                        modality.__class__.__name__,
                    )
                )
            checkpoint_dataset = checkpoint.get("dataset_name")
            if checkpoint_dataset and checkpoint_dataset != self.dataset_name:
                raise RuntimeError(
                    "Attribute checkpoint dataset {} does not match {}.".format(
                        checkpoint_dataset,
                        self.dataset_name,
                    )
                )
            checkpoint_input_dim = checkpoint.get("input_embedding_dim")
            current_input_dim = int(modality.attribute_features.shape[1])
            if (
                checkpoint_input_dim is not None
                and int(checkpoint_input_dim) != current_input_dim
            ):
                raise RuntimeError(
                    "Attribute checkpoint input dimension {} does not match {}.".format(
                        checkpoint_input_dim,
                        current_input_dim,
                    )
                )
            checkpoint_representation = checkpoint.get("representation")
            current_representation = (modality.attribute_metadata or {}).get(
                "representation"
            )
            if (
                checkpoint_representation
                and current_representation
                and checkpoint_representation != current_representation
            ):
                raise RuntimeError(
                    "Attribute checkpoint representation {} does not match {}.".format(
                        checkpoint_representation,
                        current_representation,
                    )
                )

        state_dict = self._extract_state_dict(checkpoint)
        if not state_dict:
            raise RuntimeError("Attribute checkpoint contains an empty state_dict.")
        clear_cache = getattr(modality, "clear_frozen_similarity_cache", None)
        if clear_cache is not None:
            clear_cache()
        modality.projector.load_state_dict(state_dict, strict=True)
        return checkpoint

    def _visual_checkpoint_payload(self):
        if "visual" not in self.alignment_modalities:
            raise RuntimeError(
                "Cannot save visual checkpoint: visual modality is inactive."
            )
        modality = self.alignment_modalities["visual"]
        if not modality.supports_training():
            raise RuntimeError(
                "Cannot save visual checkpoint without enabling the visual projector."
            )
        return {
            "checkpoint_version": self.CHECKPOINT_VERSION,
            "dataset_name": self.dataset_name,
            "active_modules": list(self.active_modules),
            "module_name": "visual",
            "module_class": modality.__class__.__name__,
            "modality_name": modality.modality_name,
            "state_dict": modality.projector.state_dict(),
            "visual_input_path": modality.visual_input_path,
            "input_embedding_dim": int(modality.visual_embeddings.shape[1]),
            "metadata": modality.visual_metadata,
        }

    def save_visual_model(self, checkpoint_path):
        torch.save(self._visual_checkpoint_payload(), checkpoint_path)
        return checkpoint_path

    def load_visual_model(self, checkpoint_path, map_location=None):
        if "visual" not in self.alignment_modalities:
            raise RuntimeError(
                "Cannot load visual checkpoint without enabling the visual module."
            )
        modality = self.alignment_modalities["visual"]
        if not modality.supports_training():
            raise RuntimeError(
                "Cannot load visual checkpoint without enabling the visual projector."
            )

        checkpoint = torch.load(
            checkpoint_path,
            map_location=map_location,
            weights_only=True,
        )
        if isinstance(checkpoint, dict):
            checkpoint_module_name = checkpoint.get("module_name")
            if checkpoint_module_name and checkpoint_module_name != "visual":
                raise RuntimeError(
                    "Checkpoint module {} is not a visual checkpoint.".format(
                        checkpoint_module_name
                    )
                )
            checkpoint_dataset = checkpoint.get("dataset_name")
            if checkpoint_dataset and checkpoint_dataset != self.dataset_name:
                raise RuntimeError(
                    "Visual checkpoint dataset {} does not match {}.".format(
                        checkpoint_dataset,
                        self.dataset_name,
                    )
                )
            checkpoint_input_dim = checkpoint.get("input_embedding_dim")
            current_input_dim = int(modality.visual_embeddings.shape[1])
            if (
                checkpoint_input_dim is not None
                and int(checkpoint_input_dim) != current_input_dim
            ):
                raise RuntimeError(
                    "Visual checkpoint input dimension {} does not match {}.".format(
                        checkpoint_input_dim,
                        current_input_dim,
                    )
                )

        state_dict = self._extract_state_dict(checkpoint)
        if not state_dict:
            raise RuntimeError("Visual checkpoint contains an empty state_dict.")
        modality.clear_frozen_similarity_cache()
        modality.projector.load_state_dict(state_dict, strict=True)
        return checkpoint

    def save_decision_fusion_model(self, checkpoint_path):
        if not self.has_decision_fusion():
            raise RuntimeError("Cannot save Dynamic Decision Fusion checkpoint: Dynamic Decision Fusion is inactive.")
        torch.save(
            self._checkpoint_payload("decision_fusion", self.decision_fusion),
            checkpoint_path,
        )
        return checkpoint_path

    def load_decision_fusion_model(self, checkpoint_path, map_location=None):
        if not self.has_decision_fusion():
            raise RuntimeError(
                "Cannot load Dynamic Decision Fusion checkpoint: Dynamic Decision Fusion is inactive."
            )
        checkpoint = torch.load(
            checkpoint_path,
            map_location=map_location,
            weights_only=True,
        )
        if isinstance(checkpoint, dict):
            checkpoint_class = checkpoint.get("module_class")
            if checkpoint_class and checkpoint_class != self.decision_fusion.__class__.__name__:
                raise RuntimeError(
                    "Fusion checkpoint class {} does not match active Dynamic Decision Fusion {}.".format(
                        checkpoint_class,
                        self.decision_fusion.__class__.__name__,
                    )
                )
            checkpoint_score_order = checkpoint.get("score_order")
            if (
                checkpoint_score_order
                and checkpoint_score_order != list(self.decision_fusion.SCORE_ORDER)
            ):
                raise RuntimeError(
                    "Fusion checkpoint score order {} does not match {}.".format(
                        checkpoint_score_order,
                        list(self.decision_fusion.SCORE_ORDER),
                    )
                )
        state_dict = self._extract_state_dict(checkpoint)
        self.decision_fusion.load_state_dict(state_dict, strict=True)
        return checkpoint

    def encode_graph_features(self, dataset_state):
        if not self.has_graph():
            return None
        return self.graph_modality.encode(dataset_state)

    def compute_graph_similarity(
        self,
        dataset_state,
        source_entity_ids,
        target_entity_ids,
        graph_embeddings=None,
        output_device=None,
    ):
        if not self.has_graph():
            return None
        return self.graph_modality.compute_similarity(
            dataset_state=dataset_state,
            source_entity_ids=source_entity_ids,
            target_entity_ids=target_entity_ids,
            graph_embeddings=graph_embeddings,
            output_device=output_device,
        )

    def get_graph_layer_weight_summary(self):
        if not self.has_graph():
            return None
        return self.graph_modality.get_layer_weight_summary()

    def get_decision_fusion_weight_summary(self):
        return self.decision_fusion.get_weight_summary()

    def compute_training_loss(self, train_pairs, dataset_state):
        losses = []
        decision_score_tensors = {}
        decision_validity_masks = {}
        collect_decision_scores = self.decision_fusion_training_enabled
        compute_base_losses = not self.decision_fusion_only_training

        if self.has_graph():
            graph_loss, graph_scores = (
                self.graph_modality.compute_alignment_loss_and_scores(
                    loss_module=self.loss_module,
                    train_pairs=train_pairs,
                    dataset_state=dataset_state,
                    return_score_matrix=collect_decision_scores,
                    compute_loss=compute_base_losses,
                )
            )
            if graph_loss is not None:
                losses.append(graph_loss)
            if graph_scores is not None:
                decision_score_tensors["graph"] = graph_scores

        (
            modality_loss,
            modality_score_tensors,
            modality_validity_masks,
        ) = self.compute_modality_training_outputs(
            train_pairs=train_pairs,
            dataset_state=dataset_state,
            return_score_matrices=collect_decision_scores,
            compute_losses=compute_base_losses,
        )
        if modality_loss is not None:
            losses.append(modality_loss)
        decision_score_tensors.update(modality_score_tensors)
        decision_validity_masks.update(modality_validity_masks)

        decision_fusion_loss = self.compute_decision_fusion_training_loss(
            train_pairs=train_pairs,
            score_tensors=decision_score_tensors,
            validity_masks=decision_validity_masks,
        )
        if decision_fusion_loss is not None:
            losses.append(self.decision_fusion_loss_weight * decision_fusion_loss)

        if not losses:
            raise RuntimeError(
                "Training loss requested, but no active modality provides a trainable loss."
            )

        total_loss = losses[0]
        for loss in losses[1:]:
            total_loss = total_loss + loss
        return total_loss

    def compute_modality_training_outputs(
        self,
        train_pairs,
        dataset_state,
        return_score_matrices=False,
        compute_losses=True,
    ):
        modality_losses = []
        score_tensors = {}
        validity_masks = {}
        for module_name, modality in self.alignment_modalities.items():
            compute_outputs = getattr(modality, "compute_training_loss_and_scores", None)
            if compute_outputs is None:
                loss = (
                    modality.compute_training_loss(train_pairs, dataset_state)
                    if compute_losses
                    else None
                )
                score_tensor = None
                validity_mask = None
            else:
                loss, score_tensor, validity_mask = compute_outputs(
                    train_pairs=train_pairs,
                    dataset_state=dataset_state,
                    return_score_matrix=return_score_matrices,
                    compute_loss=compute_losses,
                )
            if loss is not None:
                modality_losses.append(loss)
            if score_tensor is not None:
                score_tensors[module_name] = score_tensor
            if validity_mask is not None:
                validity_masks[module_name] = validity_mask

        if not modality_losses:
            modality_loss = None
        else:
            modality_loss = modality_losses[0]
            for loss in modality_losses[1:]:
                modality_loss = modality_loss + loss
        return modality_loss, score_tensors, validity_masks

    def validity_aware_equal_decision_weights(
        self,
        weights,
        score_tensors,
        validity_masks,
    ):
        """Average each modality only over candidate pairs where it is valid."""
        reference = weights[..., 0]
        validity_parts = []
        for module_name in self.decision_fusion.SCORE_ORDER:
            score = score_tensors.get(module_name)
            if score is None:
                validity_parts.append(
                    torch.zeros_like(reference, dtype=torch.bool)
                )
                continue
            validity = validity_masks.get(module_name)
            if validity is None:
                validity = torch.ones_like(score, dtype=torch.bool)
            validity_parts.append(
                validity.to(device=weights.device, dtype=torch.bool)
            )
        validity_tensor = torch.stack(validity_parts, dim=-1)
        validity_float = validity_tensor.to(weights.dtype)
        valid_counts = validity_float.sum(dim=(0, 1))
        mean_weights = (
            (weights * validity_float).sum(dim=(0, 1))
            / valid_counts.clamp_min(1.0)
        )
        return mean_weights, valid_counts

    def compute_decision_fusion_training_loss(self, train_pairs, score_tensors, validity_masks):
        if not self.decision_fusion_training_enabled or len(score_tensors) < 2:
            return None

        prepared_scores = {
            module_name: (
                score.detach() if self.decision_fusion_detach_scores else score
            )
            for module_name, score in score_tensors.items()
        }
        graph_scores = prepared_scores.pop("graph", None)
        fused_scores, weights = self.decision_fusion.fuse(
            graph_scores=graph_scores,
            modality_score_tensors=prepared_scores,
            modality_validity_masks=validity_masks,
            return_weights=True,
        )

        if fused_scores.shape[0] < 2:
            return fused_scores.sum() * 0.0

        labels = torch.arange(fused_scores.shape[0], device=fused_scores.device)
        scaled_scores = fused_scores / self.decision_score_temperature
        left_to_right_loss = F.cross_entropy(
            scaled_scores,
            labels,
            reduction="none",
        )
        right_to_left_loss = F.cross_entropy(
            scaled_scores.t(),
            labels,
            reduction="none",
        )
        pair_loss = 0.5 * (left_to_right_loss + right_to_left_loss)

        decision_fusion_loss = pair_loss.mean()

        if self.decision_fusion_balance_weight > 0:
            mean_weights, valid_counts = (
                self.validity_aware_equal_decision_weights(
                    weights=weights,
                    score_tensors=score_tensors,
                    validity_masks=validity_masks,
                )
            )
            active_mask = torch.as_tensor(
                [
                    module_name in score_tensors
                    for module_name in self.decision_fusion.SCORE_ORDER
                ],
                dtype=torch.bool,
                device=mean_weights.device,
            ) & (valid_counts > 0)
            active_weights = mean_weights[active_mask]
            active_weights = active_weights / active_weights.sum().clamp_min(1e-12)
            uniform_weight = 1.0 / float(active_weights.numel())
            balance_loss = torch.sum(
                active_weights
                * (
                    torch.log(active_weights.clamp_min(1e-12))
                    - torch.log(
                        torch.as_tensor(
                            uniform_weight,
                            dtype=active_weights.dtype,
                            device=active_weights.device,
                        )
                    )
                )
            )
            decision_fusion_loss = decision_fusion_loss + self.decision_fusion_balance_weight * balance_loss
        if getattr(self, "decision_fusion_anchor_weight", 0.0) > 0:
            decision_fusion_loss = decision_fusion_loss + (
                self.decision_fusion_anchor_weight * self.compute_decision_fusion_anchor_loss()
            )
        return decision_fusion_loss

    def get_modality_score_tensors(
        self,
        row_indices=None,
        col_indices=None,
        device=None,
        return_validity=False,
    ):
        score_tensors = {}
        validity_masks = {}
        for module_name, modality in self.alignment_modalities.items():
            score_output = modality.get_score_tensor(
                row_indices=row_indices,
                col_indices=col_indices,
                device=device,
                return_validity=return_validity,
            )
            if return_validity:
                score_tensor, validity_mask = score_output
                score_tensors[module_name] = score_tensor
                validity_masks[module_name] = validity_mask
            else:
                score_tensors[module_name] = score_output
        if return_validity:
            return score_tensors, validity_masks
        return score_tensors

    def compute_fused_similarity(self, graph_scores=None, row_indices=None, col_indices=None):
        if row_indices is None and col_indices is None:
            return self._compute_chunked_fused_similarity(
                graph_scores=graph_scores,
            )

        device = graph_scores.device if graph_scores is not None else None
        modality_score_tensors, modality_validity_masks = self.get_modality_score_tensors(
            row_indices=row_indices,
            col_indices=col_indices,
            device=device,
            return_validity=True,
        )
        return self.decision_fusion.fuse(
            graph_scores=graph_scores,
            modality_score_tensors=modality_score_tensors,
            modality_validity_masks=modality_validity_masks,
        )

    def _evaluation_score_shape(self, graph_scores):
        if graph_scores is not None:
            return tuple(graph_scores.shape)
        for modality in self.alignment_modalities.values():
            source_ids = getattr(modality, "eval_source_entity_ids", None)
            target_ids = getattr(modality, "eval_target_entity_ids", None)
            if source_ids is not None and target_ids is not None:
                return int(source_ids.numel()), int(target_ids.numel())
        raise ValueError("Unable to determine the evaluation score-matrix shape.")

    def _compute_chunked_fused_similarity(self, graph_scores=None):
        row_count, col_count = self._evaluation_score_shape(graph_scores)
        fused_scores = None
        default_chunk_size = getattr(
            getattr(self, "args", None),
            "decision_fusion_chunk_size",
            row_count,
        )
        chunk_size = int(
            getattr(self.decision_fusion, "chunk_size", default_chunk_size)
        )
        chunk_size = max(1, chunk_size)
        self.decision_fusion.start_weight_summary_accumulation()
        try:
            for start in range(0, row_count, chunk_size):
                end = min(start + chunk_size, row_count)
                chunk_row_indices = list(range(start, end))
                graph_chunk = (
                    graph_scores[start:end]
                    if graph_scores is not None
                    else None
                )
                device = (
                    graph_chunk.device
                    if graph_chunk is not None
                    else None
                )
                modality_scores, modality_masks = self.get_modality_score_tensors(
                    row_indices=chunk_row_indices,
                    col_indices=None,
                    device=device,
                    return_validity=True,
                )
                fused_chunk = self.decision_fusion.fuse(
                    graph_scores=graph_chunk,
                    modality_score_tensors=modality_scores,
                    modality_validity_masks=modality_masks,
                )
                if fused_scores is None:
                    fused_scores = torch.empty(
                        (row_count, col_count),
                        dtype=fused_chunk.dtype,
                        device=fused_chunk.device,
                    )
                fused_scores[start:end] = fused_chunk
        finally:
            self.decision_fusion.finish_weight_summary_accumulation()

        if fused_scores is None:
            raise ValueError("No active modules selected for similarity computation.")
        return fused_scores
