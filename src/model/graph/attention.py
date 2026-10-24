import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_sum


class DimensionWiseMessageFilteringGraphAttention(nn.Module):
    """Relation-aware GAT with dimension-wise filtering and layer weighting."""

    def __init__(
        self,
        node_dim,
        depth=1,
        device=None,
        message_filter_hidden_dim=None,
        message_filter_input="embeddings",
        message_filter_scale=2.0,
        use_relation_reflection=True,
        layer_weight_mode="learned",
        fixed_layer_weights=None,
        layer_prior_type="uniform",
        layer_prior_loc=None,
        layer_prior_scale=1.0,
        layer_prior_rate=None,
    ):
        super().__init__()
        self.node_dim = node_dim
        self.activation = torch.nn.Tanh()
        self.depth = depth
        self.device = device
        self.message_filter_input = message_filter_input
        self.message_filter_scale = message_filter_scale
        self.use_relation_reflection = use_relation_reflection
        self.layer_weight_mode = layer_weight_mode
        self.fixed_layer_weights = None
        self.layer_prior_type = layer_prior_type
        self.layer_prior_loc = float(depth if layer_prior_loc is None else layer_prior_loc)
        self.layer_prior_scale = float(layer_prior_scale)
        self.layer_prior_rate = float(depth if layer_prior_rate is None else layer_prior_rate)
        self.relation_attention_kernels = nn.ParameterList()
        self.message_filter_networks = nn.ModuleList()

        if message_filter_hidden_dim is None:
            message_filter_hidden_dim = node_dim
        self.message_filter_hidden_dim = message_filter_hidden_dim

        for _ in range(self.depth):
            relation_attention_kernel = torch.nn.Parameter(
                data=torch.empty(self.node_dim, 1, dtype=torch.float32)
            )
            torch.nn.init.xavier_uniform_(relation_attention_kernel)
            self.relation_attention_kernels.append(relation_attention_kernel)
            if self.message_filter_input in {"embeddings", "input_features"}:
                self.message_filter_networks.append(
                    nn.Sequential(
                        nn.Linear(self.node_dim, self.message_filter_hidden_dim),
                        nn.Tanh(),
                        nn.Linear(self.message_filter_hidden_dim, self.node_dim),
                    )
                )

        if self.layer_weight_mode == "learned":
            self.layer_importance_logits = nn.Parameter(torch.zeros(self.depth + 1))
        elif self.layer_weight_mode in {"uniform", "fixed", "none"}:
            self.register_parameter("layer_importance_logits", None)
        else:
            raise ValueError(
                "Unsupported propagation-layer weight mode: {}".format(layer_weight_mode)
            )

        if self.layer_weight_mode == "fixed":
            if fixed_layer_weights is None or len(fixed_layer_weights) != self.depth + 1:
                raise ValueError(
                    "Expected {} fixed propagation-layer weights, received {}.".format(
                        self.depth + 1,
                        0 if fixed_layer_weights is None else len(fixed_layer_weights),
                    )
                )
            fixed_weights = torch.as_tensor(fixed_layer_weights, dtype=torch.float32)
            if not torch.isfinite(fixed_weights).all() or fixed_weights.sum() <= 0:
                raise ValueError("Fixed propagation-layer weights must be finite with a positive sum.")
            self.fixed_layer_weights = fixed_weights / fixed_weights.sum()

        if self.layer_prior_type not in {"uniform", "folded_normal", "poisson"}:
            raise ValueError(
                "Unsupported propagation-layer prior type: {}".format(layer_prior_type)
            )

    def layer_weights(self, feature_device):
        if self.layer_weight_mode == "learned":
            return torch.softmax(self.layer_importance_logits, dim=0).to(feature_device)
        if self.layer_weight_mode == "uniform":
            return torch.full(
                (self.depth + 1,),
                1.0 / (self.depth + 1),
                dtype=torch.float32,
                device=feature_device,
            )
        if self.layer_weight_mode == "fixed":
            return self.fixed_layer_weights.to(feature_device)
        return torch.ones(self.depth + 1, dtype=torch.float32, device=feature_device)

    def layer_prior_probabilities(self, feature_device):
        layer_ids = torch.arange(
            self.depth + 1,
            dtype=torch.float32,
            device=feature_device,
        )
        if self.layer_prior_type == "uniform":
            logits = torch.zeros_like(layer_ids)
        elif self.layer_prior_type == "folded_normal":
            scale = max(self.layer_prior_scale, 1e-6)
            loc = torch.as_tensor(
                self.layer_prior_loc,
                dtype=torch.float32,
                device=feature_device,
            )
            positive_log_prob = -0.5 * torch.square((layer_ids - loc) / scale)
            negative_log_prob = -0.5 * torch.square((-layer_ids - loc) / scale)
            logits = torch.logaddexp(positive_log_prob, negative_log_prob)
        else:
            rate = max(self.layer_prior_rate, 1e-6)
            rate_tensor = torch.as_tensor(
                rate,
                dtype=torch.float32,
                device=feature_device,
            )
            logits = layer_ids * torch.log(rate_tensor) - rate_tensor - torch.lgamma(
                layer_ids + 1.0
            )
        return torch.softmax(logits, dim=0)

    def layer_prior_regularization(self):
        if self.layer_weight_mode != "learned":
            if self.layer_importance_logits is not None:
                return self.layer_importance_logits.sum() * 0.0
            return None
        q_probs = torch.softmax(self.layer_importance_logits, dim=0)
        prior_probs = self.layer_prior_probabilities(self.layer_importance_logits.device)
        return torch.sum(
            q_probs
            * (
                torch.log(q_probs.clamp_min(1e-12))
                - torch.log(prior_probs.clamp_min(1e-12))
            )
        )

    def forward(
        self,
        inputs,
        return_layers=False,
        apply_layer_weights=True,
        apply_message_filter=True,
        return_diagnostics=False,
    ):
        outputs = []
        diagnostics = []
        features = inputs[0]
        rel_emb = inputs[1]
        entity_adjacency_edges = inputs[2]
        edge_relation_indices = inputs[3]
        edge_relation_values = inputs[4]
        edge_count = inputs[5]
        relation_feature_dim = inputs[6]
        entity_count = inputs[7]
        mask = inputs[8]
        features = self.activation(features)
        first_features = features
        layer_weights = self.layer_weights(features.device)

        has_image_features = len(inputs) > 9 and inputs[9] is not None
        precomputed_edge_relation_features = (
            inputs[10] if len(inputs) > 10 else None
        )
        if has_image_features:
            img_feat = inputs[9]
            img_feat = self.activation(img_feat)

        outputs.append(features)

        for layer_index in range(self.depth):
            attention_kernel = self.relation_attention_kernels[layer_index]

            neighbor_features = features[entity_adjacency_edges[1, :].long()]
            message_filter = None
            if self.message_filter_input == "constant" and apply_message_filter:
                message_filter = torch.full_like(features, self.message_filter_scale)
                neighbor_features = neighbor_features * message_filter[
                    entity_adjacency_edges[1, :].long()
                ]
            elif self.message_filter_input is not None and apply_message_filter:
                if self.message_filter_input == "input_features":
                    filter_state = first_features
                elif self.message_filter_input == "embeddings":
                    filter_state = features
                else:
                    raise ValueError(
                        "Unsupported dimension-wise message filter type: {}".format(
                            self.message_filter_input
                        )
                    )
                message_filter = torch.sigmoid(
                    self.message_filter_networks[layer_index](filter_state)
                )
                neighbor_features = (
                    neighbor_features
                    * self.message_filter_scale
                    * message_filter[entity_adjacency_edges[1, :].long()]
                )

            if mask is not None:
                edge_mask_weights = torch.zeros(entity_adjacency_edges.shape[1]).to(
                    self.device
                )
                masked_relation_values = (mask * edge_relation_values).float()
                edge_mask_weights = edge_mask_weights.scatter_add_(
                    0,
                    edge_relation_indices[0].long(),
                    masked_relation_values,
                )
            if precomputed_edge_relation_features is None:
                edge_relation_matrix = torch.sparse_coo_tensor(
                    indices=edge_relation_indices,
                    values=edge_relation_values,
                    size=[edge_count, relation_feature_dim],
                    dtype=torch.float32,
                )
                if has_image_features:
                    edge_relation_features = torch.sparse.mm(
                        edge_relation_matrix,
                        img_feat,
                    )
                else:
                    edge_relation_features = torch.sparse.mm(
                        edge_relation_matrix,
                        rel_emb,
                    )
                edge_relation_features = F.normalize(
                    edge_relation_features,
                    dim=1,
                    p=2,
                )
            else:
                edge_relation_features = precomputed_edge_relation_features
            if self.use_relation_reflection:
                neighbor_features = (
                    neighbor_features
                    - 2
                    * torch.sum(
                        neighbor_features * edge_relation_features,
                        dim=1,
                        keepdim=True,
                    )
                    * edge_relation_features
                )

            attention_scores = torch.squeeze(
                torch.mm(edge_relation_features, attention_kernel),
                dim=-1,
            )

            if mask is not None:
                attention_scores[edge_mask_weights == 0] = -1e10

            attention_matrix = torch.sparse_coo_tensor(
                indices=entity_adjacency_edges,
                values=attention_scores,
                size=[entity_count, entity_count],
            )
            attention_matrix = torch.sparse.softmax(attention_matrix, dim=1)

            if mask is not None:
                mask_adjacency = torch.sparse_coo_tensor(
                    indices=entity_adjacency_edges,
                    values=edge_mask_weights,
                    size=[entity_count, entity_count],
                )
                attention_matrix = attention_matrix * mask_adjacency

            new_features = scatter_sum(
                src=neighbor_features
                * torch.unsqueeze(attention_matrix.coalesce().values(), dim=-1),
                dim=0,
                index=entity_adjacency_edges[0, :].long(),
                dim_size=entity_count,
            )
            features = self.activation(new_features)
            outputs.append(features)
            if return_diagnostics:
                edge_values = attention_matrix.coalesce().values()
                if message_filter is None:
                    filter_strength = torch.ones(
                        edge_values.shape[0],
                        dtype=edge_values.dtype,
                        device=edge_values.device,
                    )
                else:
                    filter_strength = message_filter[
                        entity_adjacency_edges[1, :].long()
                    ].mean(dim=-1)
                coalesced_indices = attention_matrix.coalesce().indices()
                diagnostics.append(
                    {
                        "edge_index": coalesced_indices.detach(),
                        "attention": edge_values.detach(),
                        "filter_strength": filter_strength.detach(),
                        "filter_weights": (
                            message_filter.detach()
                            if message_filter is not None
                            else torch.ones_like(features).detach()
                        ),
                    }
                )

        if return_layers:
            if return_diagnostics:
                return outputs, diagnostics
            return outputs

        if apply_layer_weights:
            outputs = [
                output * layer_weights[layer_index]
                for layer_index, output in enumerate(outputs)
            ]
        outputs = torch.cat(outputs, dim=-1)
        return outputs
