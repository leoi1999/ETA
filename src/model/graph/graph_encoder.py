import torch
import torch.nn as nn

from .attention import (
    DimensionWiseMessageFilteringGraphAttention,
)


class GraphEncoder(nn.Module):
    def __init__(
        self,
        node_hidden,
        rel_hidden,
        device,
        entity_count,
        relation_feature_dim,
        dropout_rate=0.0,
        depth=2,
        message_filter_hidden_dim=None,
        message_filter_input="embeddings",
        message_filter_scale=2.0,
        use_relation_reflection=True,
        layer_weight_mode="learned",
        layer_fixed_weights=None,
        layer_prior_type="uniform",
        layer_prior_loc=None,
        layer_prior_scale=1.0,
        layer_prior_rate=None,
    ):
        super().__init__()
        self.node_hidden = node_hidden
        self.depth = depth
        self.device = device
        self.dropout = nn.Dropout(dropout_rate)
        self.entity_embedding = nn.Embedding(entity_count, node_hidden)
        nn.init.xavier_uniform_(self.entity_embedding.weight)
        self.relation_embedding = nn.Embedding(relation_feature_dim, rel_hidden)
        nn.init.xavier_uniform_(self.relation_embedding.weight)
        self._static_sparse_cache = {}
        encoder_cls = DimensionWiseMessageFilteringGraphAttention

        encoder_kwargs = {
            "node_dim": self.node_hidden,
            "depth": self.depth,
            "device": self.device,
        }
        if encoder_cls is DimensionWiseMessageFilteringGraphAttention:
            encoder_kwargs.update(
                {
                    "message_filter_hidden_dim": message_filter_hidden_dim,
                    "message_filter_input": message_filter_input,
                    "message_filter_scale": message_filter_scale,
                    "use_relation_reflection": use_relation_reflection,
                    "layer_weight_mode": layer_weight_mode,
                    "fixed_layer_weights": layer_fixed_weights,
                    "layer_prior_type": layer_prior_type,
                    "layer_prior_loc": layer_prior_loc,
                    "layer_prior_scale": layer_prior_scale,
                    "layer_prior_rate": layer_prior_rate,
                }
            )
        self.entity_encoder = DimensionWiseMessageFilteringGraphAttention(**encoder_kwargs)
        self.relation_encoder = DimensionWiseMessageFilteringGraphAttention(**encoder_kwargs)
        if (
            layer_weight_mode == "learned"
        ):
            # In adaptive-layer training both views consume the same layer
            # loss and the same prior. Separate zero-initialized logits
            # therefore receive identical gradients forever. Share the actual
            # Parameter so entity and relation views learn one shared layer
            # importance distribution.
            self.relation_encoder.layer_importance_logits = self.entity_encoder.layer_importance_logits

    def layer_prior_regularization(self):
        if (
            getattr(self.entity_encoder, "layer_importance_logits", None) is not None
            and self.entity_encoder.layer_importance_logits
            is getattr(self.relation_encoder, "layer_importance_logits", None)
        ):
            return self.entity_encoder.layer_prior_regularization()

        regularizers = []
        for encoder in (self.entity_encoder, self.relation_encoder):
            if not hasattr(encoder, "layer_prior_regularization"):
                continue
            regularizer = encoder.layer_prior_regularization()
            if regularizer is not None:
                regularizers.append(regularizer)

        if not regularizers:
            return None

        total_regularizer = regularizers[0]
        for regularizer in regularizers[1:]:
            total_regularizer = total_regularizer + regularizer
        return total_regularizer / len(regularizers)

    def adaptive_layer_weights(self, device):
        if (
            getattr(self.entity_encoder, "layer_importance_logits", None) is not None
            and self.entity_encoder.layer_importance_logits
            is getattr(self.relation_encoder, "layer_importance_logits", None)
        ):
            return self.entity_encoder.layer_weights(device)

        layer_weights = []
        for encoder in (self.entity_encoder, self.relation_encoder):
            if not hasattr(encoder, "layer_weights"):
                continue
            layer_weights.append(encoder.layer_weights(device))

        if not layer_weights:
            return None

        combined_weights = layer_weights[0]
        for weights in layer_weights[1:]:
            combined_weights = combined_weights + weights
        combined_weights = combined_weights / len(layer_weights)
        return combined_weights / combined_weights.sum().clamp_min(1e-12)

    @staticmethod
    def _sparse_cache_key(edge_indices, row_count, column_count):
        return (
            edge_indices.data_ptr(),
            tuple(edge_indices.shape),
            str(edge_indices.device),
            int(row_count),
            int(column_count),
        )

    def _row_normalized_sparse_edges(
        self,
        cache_name,
        edge_indices,
        row_count,
        column_count,
    ):
        cache_key = self._sparse_cache_key(edge_indices, row_count, column_count)
        cached = self._static_sparse_cache.get(cache_name)
        if cached is not None and cached[0] == cache_key:
            return cached[1]

        row_indices = edge_indices[0].long()
        row_counts = torch.zeros(
            row_count,
            dtype=torch.float32,
            device=edge_indices.device,
        )
        row_counts.scatter_add_(
            0,
            row_indices,
            torch.ones_like(row_indices, dtype=torch.float32),
        )
        values = row_counts[row_indices].clamp_min(1.0).reciprocal()
        sparse_edges = torch.sparse_coo_tensor(
            indices=edge_indices,
            values=values,
            size=(row_count, column_count),
            device=edge_indices.device,
        ).coalesce()
        self._static_sparse_cache[cache_name] = (cache_key, sparse_edges)
        return sparse_edges

    def average_neighbor_embeddings(
        self,
        cache_name,
        edge_indices,
        embeddings,
        column_count,
        row_count,
    ):
        sparse_edges = self._row_normalized_sparse_edges(
            cache_name=cache_name,
            edge_indices=edge_indices,
            row_count=row_count,
            column_count=column_count,
        )
        return torch.sparse.mm(sparse_edges, embeddings)

    def _edge_relation_sparse_matrix(
        self,
        edge_relation_indices,
        edge_relation_values,
        edge_count,
        relation_feature_dim,
    ):
        cache_key = (
            edge_relation_indices.data_ptr(),
            edge_relation_values.data_ptr(),
            tuple(edge_relation_indices.shape),
            str(edge_relation_indices.device),
            int(edge_count),
            int(relation_feature_dim),
        )
        cached = self._static_sparse_cache.get("edge_relation")
        if cached is not None and cached[0] == cache_key:
            return cached[1]

        sparse_matrix = torch.sparse_coo_tensor(
            indices=edge_relation_indices,
            values=edge_relation_values,
            size=(edge_count, relation_feature_dim),
            dtype=torch.float32,
            device=edge_relation_indices.device,
        ).coalesce()
        self._static_sparse_cache["edge_relation"] = (cache_key, sparse_matrix)
        return sparse_matrix

    def _preparentity_encoder_inputs(
        self,
        entity_feature_edges,
        relation_feature_edges,
        entity_count,
        relation_feature_dim,
        entity_adjacency_edges,
        edge_relation_indices,
        edge_relation_values,
        edge_count,
        mask,
    ):
        # The paper initializes the entity branch from each entity's own
        # embedding. Neighbor information enters only through GAT propagation;
        # averaging here would make layer 0 already contain one-hop features.
        del entity_feature_edges
        entity_features = self.entity_embedding.weight
        relation_features = self.average_neighbor_embeddings(
            "relation_features",
            relation_feature_edges,
            self.relation_embedding.weight,
            relation_feature_dim,
            entity_count,
        )

        edge_relation_matrix = self._edge_relation_sparse_matrix(
            edge_relation_indices=edge_relation_indices,
            edge_relation_values=edge_relation_values,
            edge_count=edge_count,
            relation_feature_dim=relation_feature_dim,
        )
        normalized_edge_relation_features = torch.nn.functional.normalize(
            torch.sparse.mm(edge_relation_matrix, self.relation_embedding.weight),
            dim=1,
            p=2,
        )

        encoder_inputs = [
            self.relation_embedding.weight,
            entity_adjacency_edges,
            edge_relation_indices,
            edge_relation_values,
            edge_count,
            relation_feature_dim,
            entity_count,
            mask,
            None,
            normalized_edge_relation_features,
        ]
        return entity_features, relation_features, encoder_inputs

    def encode_layers(
        self,
        entity_feature_edges,
        relation_feature_edges,
        entity_count,
        relation_feature_dim,
        entity_adjacency_edges,
        edge_relation_indices,
        edge_relation_values,
        edge_count,
        mask,
        apply_message_filter=True,
        return_diagnostics=False,
    ):
        entity_features, relation_features, encoder_inputs = (
            self._preparentity_encoder_inputs(
                entity_feature_edges,
                relation_feature_edges,
                entity_count,
                relation_feature_dim,
                entity_adjacency_edges,
                edge_relation_indices,
                edge_relation_values,
                edge_count,
                mask,
            )
        )
        entity_output = self.entity_encoder(
            [entity_features] + encoder_inputs,
            return_layers=True,
            apply_message_filter=apply_message_filter,
            return_diagnostics=return_diagnostics,
        )
        if return_diagnostics:
            entity_layers, entity_diagnostics = entity_output
        else:
            entity_layers = entity_output
        relation_layers = self.relation_encoder(
            [relation_features] + encoder_inputs,
            return_layers=True,
            apply_message_filter=apply_message_filter,
        )
        combined_layers = [
            self.dropout(torch.cat([entity_layer, relation_layer], dim=-1))
            for entity_layer, relation_layer in zip(entity_layers, relation_layers)
        ]
        if return_diagnostics:
            return combined_layers, entity_diagnostics
        return combined_layers

    def forward(
        self,
        entity_feature_edges,
        relation_feature_edges,
        entity_count,
        relation_feature_dim,
        entity_adjacency_edges,
        edge_relation_indices,
        edge_relation_values,
        edge_count,
        mask,
    ):
        entity_features, relation_features, encoder_inputs = (
            self._preparentity_encoder_inputs(
                entity_feature_edges,
                relation_feature_edges,
                entity_count,
                relation_feature_dim,
                entity_adjacency_edges,
                edge_relation_indices,
                edge_relation_values,
                edge_count,
                mask,
            )
        )
        entity_encoded = self.entity_encoder([entity_features] + encoder_inputs)
        relation_encoded = self.relation_encoder([relation_features] + encoder_inputs)
        graph_embeddings = torch.cat([entity_encoded, relation_encoded], dim=-1)

        graph_embeddings = self.dropout(graph_embeddings)
        return [graph_embeddings]
