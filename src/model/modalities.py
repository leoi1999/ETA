import time

import torch
import torch.nn as nn

from .attribute.module import AttributeModalityModule
from .graph import GraphEncoder
from .visual.module import VisualModalityModule


SUPPORTED_MODULES = ("graph", "attribute", "visual")


def parse_active_modules(modules_config):
    raw_modules = [item.strip().lower() for item in modules_config.split(",") if item.strip()]
    modules = []
    for module in raw_modules:
        if module not in SUPPORTED_MODULES:
            raise ValueError("Unsupported module: {}".format(module))
        if module not in modules:
            modules.append(module)
    return modules


class GraphModalityModule(nn.Module):
    def __init__(self, args, dataset_state):
        super().__init__()
        self.module_name = "graph"
        self.modality_name = "eta_graph"
        self.layer_regularization_weight = args.layer_regularization_weight
        self.encoder = GraphEncoder(
            node_hidden=args.node_dim,
            rel_hidden=args.relation_dim,
            entity_count=dataset_state.entity_count,
            relation_feature_dim=dataset_state.relation_feature_dim,
            device=args.device,
            dropout_rate=args.dropout,
            depth=args.depth,
            message_filter_hidden_dim=args.message_filter_hidden_dim,
            message_filter_input=(
                None if args.message_filter_input == "none" else args.message_filter_input
            ),
            message_filter_scale=args.message_filter_scale,
            use_relation_reflection=getattr(args, "use_relation_reflection", True),
            layer_weight_mode=args.layer_weight_mode,
            layer_fixed_weights=args.layer_fixed_weights,
            layer_prior_type=args.layer_prior_type,
            layer_prior_loc=(
                None if args.layer_prior_loc < 0 else args.layer_prior_loc
            ),
            layer_prior_scale=args.layer_prior_scale,
            layer_prior_rate=(
                None if args.layer_prior_rate < 0 else args.layer_prior_rate
            ),
        )
        self._frozen_similarity_cache = None

    @staticmethod
    def _normalize_embeddings(embeddings):
        return embeddings / (
            torch.linalg.norm(embeddings, dim=-1, keepdim=True) + 1e-5
        )

    @torch.inference_mode()
    def prepare_frozen_similarity_cache(self, dataset_state):
        """Cache entity evidence while the graph encoder is frozen."""
        uses_adaptive_layers = (
            self.modality_name == "eta_graph"
            and self.modality_name == "eta_graph"
        )
        if uses_adaptive_layers:
            normalized_layers = tuple(
                self._normalize_embeddings(layer_embedding).detach()
                for layer_embedding in self.encode_layers(dataset_state)
            )
            layer_weights = self.encoder.adaptive_layer_weights(
                normalized_layers[0].device
            ).detach()
            self._frozen_similarity_cache = {
                "mode": "adaptive_layers",
                "embeddings": normalized_layers,
                "weights": layer_weights,
            }
            cached_bytes = sum(
                value.numel() * value.element_size()
                for value in normalized_layers
            ) + layer_weights.numel() * layer_weights.element_size()
        else:
            normalized_embeddings = self._normalize_embeddings(
                self.encode(dataset_state)
            ).detach()
            self._frozen_similarity_cache = {
                "mode": "single",
                "embeddings": normalized_embeddings,
            }
            cached_bytes = (
                normalized_embeddings.numel()
                * normalized_embeddings.element_size()
            )
        return cached_bytes

    def clear_frozen_similarity_cache(self):
        self._frozen_similarity_cache = None

    def _cached_similarity(self, source_entity_ids, target_entity_ids):
        cache = getattr(self, "_frozen_similarity_cache", None)
        if cache is None:
            return None

        embeddings = cache["embeddings"]
        reference = embeddings[0] if isinstance(embeddings, tuple) else embeddings
        left_indices = torch.as_tensor(
            source_entity_ids,
            dtype=torch.long,
            device=reference.device,
        )
        right_indices = torch.as_tensor(
            target_entity_ids,
            dtype=torch.long,
            device=reference.device,
        )
        if cache["mode"] == "single":
            return embeddings[left_indices] @ embeddings[right_indices].T

        scores = None
        for layer_weight, layer_embedding in zip(cache["weights"], embeddings):
            layer_scores = (
                layer_embedding[left_indices] @ layer_embedding[right_indices].T
            )
            weighted_scores = layer_weight * layer_scores
            scores = weighted_scores if scores is None else scores + weighted_scores
        return scores

    def encode(self, dataset_state):
        encoder_outputs = self.encoder(
            dataset_state.entity_feature_edges,
            dataset_state.relation_feature_edges,
            dataset_state.entity_count,
            dataset_state.relation_feature_dim,
            dataset_state.entity_adjacency_edges,
            dataset_state.edge_relation_indices,
            dataset_state.edge_relation_values,
            dataset_state.edge_count,
            mask=None,
        )
        return encoder_outputs[0]

    def encode_layers(
        self,
        dataset_state,
        apply_message_filter=True,
        return_diagnostics=False,
    ):
        return self.encoder.encode_layers(
            dataset_state.entity_feature_edges,
            dataset_state.relation_feature_edges,
            dataset_state.entity_count,
            dataset_state.relation_feature_dim,
            dataset_state.entity_adjacency_edges,
            dataset_state.edge_relation_indices,
            dataset_state.edge_relation_values,
            dataset_state.edge_count,
            mask=None,
            apply_message_filter=apply_message_filter,
            return_diagnostics=return_diagnostics,
        )

    @torch.inference_mode()
    def encode_counterfactual_layers(self, dataset_state, apply_message_filter=True):
        """Expose layer representations for case-study counterfactuals.

        The checkpoint and all learned parameters stay unchanged.  This only
        toggles the dimension-wise message filter at inference time.
        """
        return self.encode_layers(
            dataset_state,
            apply_message_filter=apply_message_filter,
            return_diagnostics=False,
        )

    @torch.inference_mode()
    def encode_layers_with_neighbor_diagnostics(self, dataset_state):
        """Return selected graph layers and entity-branch edge diagnostics."""
        return self.encode_layers(
            dataset_state,
            apply_message_filter=True,
            return_diagnostics=True,
        )

    def compute_layer_prior_regularization(self):
        if self.layer_regularization_weight <= 0:
            return None
        regularizer = self.encoder.layer_prior_regularization()
        if regularizer is None:
            return None
        return self.layer_regularization_weight * regularizer

    def compute_alignment_loss_and_scores(
        self,
        loss_module,
        train_pairs,
        dataset_state,
        return_score_matrix=False,
        compute_loss=True,
    ):
        if not compute_loss and return_score_matrix:
            cached_scores = self._cached_similarity(
                train_pairs[:, 0],
                train_pairs[:, 1],
            )
            if cached_scores is not None:
                return None, cached_scores

        if self.modality_name == "eta_graph":
            layer_embeddings = self.encode_layers(dataset_state)
            layer_weights = self.encoder.adaptive_layer_weights(layer_embeddings[0].device)
            total_loss = None
            score_matrix = None
            left_entity_ids = train_pairs[:, 0].long()
            right_entity_ids = train_pairs[:, 1].long()
            for layer_index, layer_embedding in enumerate(layer_embeddings):
                if compute_loss:
                    layer_loss = loss_module(train_pairs, layer_embedding)
                    weighted_layer_loss = layer_weights[layer_index] * layer_loss
                    if total_loss is None:
                        total_loss = weighted_layer_loss
                    else:
                        total_loss = total_loss + weighted_layer_loss
                if return_score_matrix:
                    layer_scores = self._normalized_similarity(
                        layer_embedding[left_entity_ids],
                        layer_embedding[right_entity_ids],
                    )
                    weighted_layer_scores = layer_weights[layer_index] * layer_scores
                    if score_matrix is None:
                        score_matrix = weighted_layer_scores
                    else:
                        score_matrix = score_matrix + weighted_layer_scores
            if compute_loss:
                layer_prior_regularization = self.compute_layer_prior_regularization()
                if layer_prior_regularization is not None:
                    total_loss = total_loss + layer_prior_regularization
            return total_loss, score_matrix

        graph_embeddings = self.encode(dataset_state)
        total_loss = None
        if compute_loss:
            total_loss = loss_module(train_pairs, graph_embeddings)
            layer_prior_regularization = self.compute_layer_prior_regularization()
            if layer_prior_regularization is not None:
                total_loss = total_loss + layer_prior_regularization
        score_matrix = None
        if return_score_matrix:
            left_entity_ids = train_pairs[:, 0].long()
            right_entity_ids = train_pairs[:, 1].long()
            score_matrix = self._normalized_similarity(
                graph_embeddings[left_entity_ids],
                graph_embeddings[right_entity_ids],
            )
        return total_loss, score_matrix

    def get_layer_weight_summary(self):
        if self.modality_name != "eta_graph":
            return None
        entity_weights = self.encoder.entity_encoder.layer_weights("cpu").detach().cpu()
        relation_weights = self.encoder.relation_encoder.layer_weights("cpu").detach().cpu()
        combined_weights = (entity_weights + relation_weights) / 2
        combined_weights = combined_weights / combined_weights.sum().clamp_min(1e-12)
        return {
            "entity": entity_weights.tolist(),
            "relation": relation_weights.tolist(),
            "combined": combined_weights.tolist(),
        }

    @staticmethod
    def _normalized_similarity(left_embeddings, right_embeddings):
        left_embeddings = GraphModalityModule._normalize_embeddings(
            left_embeddings
        )
        right_embeddings = GraphModalityModule._normalize_embeddings(
            right_embeddings
        )
        return left_embeddings @ right_embeddings.T

    @torch.inference_mode()
    def compute_similarity(
        self,
        dataset_state,
        source_entity_ids,
        target_entity_ids,
        graph_embeddings=None,
        output_device=None,
    ):
        def finish(scores):
            scores = scores.detach()
            if output_device is None:
                return scores.cpu()
            return scores.to(output_device)

        cached_scores = self._cached_similarity(
            source_entity_ids,
            target_entity_ids,
        )
        if cached_scores is not None:
            return finish(cached_scores)

        uses_adaptive_layers = getattr(self, "modality_name", None) == "eta_graph"
        if not uses_adaptive_layers:
            if graph_embeddings is None:
                graph_embeddings = self.encode(dataset_state)
            embedding_device = graph_embeddings.device
            left_indices = torch.as_tensor(
                source_entity_ids,
                dtype=torch.long,
                device=embedding_device,
            )
            right_indices = torch.as_tensor(
                target_entity_ids,
                dtype=torch.long,
                device=embedding_device,
            )
            return finish(self._normalized_similarity(
                graph_embeddings[left_indices],
                graph_embeddings[right_indices],
            ))

        layer_embeddings = self.encode_layers(dataset_state)
        layer_weights = self.encoder.adaptive_layer_weights(layer_embeddings[0].device)
        left_indices = torch.as_tensor(
            source_entity_ids,
            dtype=torch.long,
            device=layer_embeddings[0].device,
        )
        right_indices = torch.as_tensor(
            target_entity_ids,
            dtype=torch.long,
            device=layer_embeddings[0].device,
        )

        graph_scores = None
        for layer_index, layer_embedding in enumerate(layer_embeddings):
            layer_scores = self._normalized_similarity(
                layer_embedding[left_indices],
                layer_embedding[right_indices],
            )
            weighted_scores = layer_weights[layer_index] * layer_scores
            if graph_scores is None:
                graph_scores = weighted_scores
            else:
                graph_scores = graph_scores + weighted_scores
        return finish(graph_scores)

def load_alignment_modality_modules(args, dataset_state, active_modules):
    started_at = time.time()
    loaders = {
        "attribute": lambda: AttributeModalityModule(
            args=args,
            dataset_state=dataset_state,
        ),
        "visual": lambda: VisualModalityModule(
            args=args,
            dataset_state=dataset_state,
        ),
    }

    alignment_modalities = nn.ModuleDict()
    for module_name in active_modules:
        if module_name not in loaders:
            continue
        alignment_modalities[module_name] = loaders[module_name]()

    return alignment_modalities, time.time() - started_at
