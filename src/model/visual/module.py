"""Visual-modality encoder over stability-reweighted CLIP features."""

import os
import pickle

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..compatibility import install_numpy_pickle_compat
from ..embedding_cache import load_or_create_embedding_cache

MODALITY_NAME = "Visual"


class ResidualVisualProjector(nn.Module):
    """Bottleneck adapter initialized to preserve the original CLIP vector."""

    def __init__(self, input_dim, hidden_dim, bottleneck_dim, dropout):
        super().__init__()
        self.normalization = nn.LayerNorm(input_dim)
        self.adapter = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, bottleneck_dim),
            nn.GELU(),
            nn.Linear(bottleneck_dim, input_dim),
        )
        nn.init.zeros_(self.adapter[-1].weight)
        nn.init.zeros_(self.adapter[-1].bias)

    def forward(self, features):
        return features + self.adapter(self.normalization(features))


def _load_clip_embedding_table(path):
    install_numpy_pickle_compat()
    with open(path, "rb") as handle:
        data = pickle.load(handle)

    if not isinstance(data, dict) or "entity_embeddings" not in data:
        raise ValueError("Expected an image.pkl payload with entity_embeddings.")
    entity_embeddings = data["entity_embeddings"]
    metadata = data.get("metadata", {})

    if not entity_embeddings:
        raise RuntimeError("No entity image embeddings found in {}".format(path))

    max_entity_id = max(int(entity_id) for entity_id in entity_embeddings)
    sample_embedding = next(iter(entity_embeddings.values()))
    embedding_dim = int(np.asarray(sample_embedding).reshape(-1).shape[0])
    table = np.zeros((max_entity_id + 1, embedding_dim), dtype=np.float32)
    mask = np.zeros((max_entity_id + 1,), dtype=np.bool_)

    for entity_id, embedding in entity_embeddings.items():
        entity_id = int(entity_id)
        table[entity_id] = np.asarray(embedding, dtype=np.float32).reshape(-1)
        mask[entity_id] = True

    return (
        torch.as_tensor(table, dtype=torch.float32),
        torch.as_tensor(mask, dtype=torch.bool),
        metadata,
    )


class VisualModalityModule(nn.Module):
    """Visual modality with an optional projector over frozen embeddings."""

    def __init__(self, args, dataset_state):
        super().__init__()
        self.module_name = "visual"
        self.modality_name = MODALITY_NAME
        self.temperature = float(args.visual_temperature)
        self.use_projector = bool(args.visual_use_projector)

        embedding_path = self._resolve_embedding_path(args, dataset_state)
        self.visual_input_path = embedding_path
        embedding_table, embedding_mask, metadata = load_or_create_embedding_cache(
            embedding_path,
            getattr(args, "embedding_cache_dir", None),
            "visual",
            _load_clip_embedding_table,
        )
        self.visual_metadata = metadata
        self.register_buffer("visual_embeddings", embedding_table)
        self.register_buffer("visual_embedding_mask", embedding_mask)
        self._frozen_entity_cache = None
        self.register_buffer(
            "eval_source_entity_ids",
            dataset_state.dev_pairs[:, 0].long().clone(),
        )
        self.register_buffer(
            "eval_target_entity_ids",
            dataset_state.dev_pairs[:, 1].long().clone(),
        )
        if self.use_projector:
            input_dim = int(self.visual_embeddings.shape[1])
            self.projector = ResidualVisualProjector(
                input_dim=input_dim,
                hidden_dim=args.visual_hidden_dim,
                bottleneck_dim=args.visual_projection_dim,
                dropout=args.visual_dropout,
            )
        else:
            self.projector = nn.Identity()

    @staticmethod
    def _resolve_embedding_path(args, dataset_state):
        if os.path.isabs(args.visual_embedding_path):
            return args.visual_embedding_path
        return os.path.join(
            args.data_root,
            dataset_state.dataset_name,
            args.visual_embedding_path,
        )

    def supports_training(self):
        return self.use_projector

    def supports_decision_fusion_training(self):
        return True

    def _valid_entity_mask(self, entity_ids):
        if not isinstance(entity_ids, torch.Tensor):
            entity_ids = torch.as_tensor(entity_ids, dtype=torch.long)
        entity_ids = entity_ids.to(self.visual_embeddings.device).long()
        in_range = entity_ids < self.visual_embeddings.shape[0]
        valid = torch.zeros_like(in_range, dtype=torch.bool)
        if torch.any(in_range):
            valid[in_range] = self.visual_embedding_mask[entity_ids[in_range]]
        return valid

    def _lookup_visual_embeddings(self, entity_ids):
        if not isinstance(entity_ids, torch.Tensor):
            entity_ids = torch.as_tensor(entity_ids, dtype=torch.long)
        entity_ids = entity_ids.to(self.visual_embeddings.device).long()

        embeddings = torch.zeros(
            entity_ids.shape[0],
            self.visual_embeddings.shape[1],
            dtype=self.visual_embeddings.dtype,
            device=self.visual_embeddings.device,
        )
        valid = self._valid_entity_mask(entity_ids)
        if torch.any(valid):
            embeddings[valid] = self.visual_embeddings[entity_ids[valid]]
        return embeddings, valid

    @torch.inference_mode()
    def prepare_frozen_similarity_cache(self, dataset_state=None):
        del dataset_state
        entity_ids = torch.arange(
            self.visual_embeddings.shape[0],
            dtype=torch.long,
            device=self.visual_embeddings.device,
        )
        embeddings, valid = self.encode_entity_ids(entity_ids)
        self._frozen_entity_cache = (embeddings.detach(), valid.detach())
        return sum(
            value.numel() * value.element_size()
            for value in self._frozen_entity_cache
        )

    def clear_frozen_similarity_cache(self):
        self._frozen_entity_cache = None

    def _lookup_frozen_entity_cache(self, entity_ids):
        if self._frozen_entity_cache is None:
            return None
        if not isinstance(entity_ids, torch.Tensor):
            entity_ids = torch.as_tensor(entity_ids, dtype=torch.long)
        cached_embeddings, cached_validity = self._frozen_entity_cache
        entity_ids = entity_ids.to(cached_embeddings.device).long()
        embeddings = torch.zeros(
            entity_ids.shape[0],
            cached_embeddings.shape[1],
            dtype=cached_embeddings.dtype,
            device=cached_embeddings.device,
        )
        valid = torch.zeros_like(entity_ids, dtype=torch.bool)
        in_range = (entity_ids >= 0) & (entity_ids < cached_embeddings.shape[0])
        if torch.any(in_range):
            selected_ids = entity_ids[in_range]
            embeddings[in_range] = cached_embeddings[selected_ids]
            valid[in_range] = cached_validity[selected_ids]
        return embeddings, valid

    def encode_entity_ids(self, entity_ids):
        cached = self._lookup_frozen_entity_cache(entity_ids)
        if cached is None:
            visual_embeddings, valid = self._lookup_visual_embeddings(entity_ids)
            projected = F.normalize(self.projector(visual_embeddings), dim=-1)
            projected = projected * valid.unsqueeze(-1).to(projected.dtype)
        else:
            projected, valid = cached
        return projected, valid

    def compute_training_loss_and_scores(
        self,
        train_pairs,
        dataset_state,
        return_score_matrix=False,
        compute_loss=True,
    ):
        del dataset_state
        left_embeddings, left_valid = self.encode_entity_ids(train_pairs[:, 0])
        right_embeddings, right_valid = self.encode_entity_ids(train_pairs[:, 1])
        cosine_scores = torch.matmul(left_embeddings, right_embeddings.t())
        training_loss = None
        if compute_loss and self.supports_training():
            paired_valid = left_valid & right_valid
            if torch.any(paired_valid):
                valid_left = left_embeddings[paired_valid]
                valid_right = right_embeddings[paired_valid]
                logits = torch.matmul(valid_left, valid_right.t()) / self.temperature
                labels = torch.arange(logits.size(0), device=logits.device)
                loss_lr = F.cross_entropy(logits, labels)
                loss_rl = F.cross_entropy(logits.t(), labels)
                training_loss = 0.5 * (loss_lr + loss_rl)
            else:
                training_loss = next(self.projector.parameters()).sum() * 0.0

        if not return_score_matrix:
            return training_loss, None, None

        score_tensor = cosine_scores.clamp(-1.0, 1.0)
        validity_mask = left_valid.unsqueeze(1) & right_valid.unsqueeze(0)
        score_tensor = score_tensor * validity_mask.to(score_tensor.dtype)
        return training_loss, score_tensor, validity_mask

    def compute_training_loss(self, train_pairs, dataset_state):
        training_loss, _, _ = self.compute_training_loss_and_scores(
            train_pairs=train_pairs,
            dataset_state=dataset_state,
            return_score_matrix=False,
            compute_loss=True,
        )
        return training_loss

    @torch.no_grad()
    def get_score_tensor(
        self,
        row_indices=None,
        col_indices=None,
        device=None,
        return_validity=False,
    ):
        source_entity_ids = self.eval_source_entity_ids
        target_entity_ids = self.eval_target_entity_ids

        if row_indices is not None:
            source_entity_ids = source_entity_ids[row_indices]
        if col_indices is not None:
            target_entity_ids = target_entity_ids[col_indices]

        left_embeddings, left_valid = self.encode_entity_ids(source_entity_ids)
        right_embeddings, right_valid = self.encode_entity_ids(target_entity_ids)
        score_tensor = torch.matmul(left_embeddings, right_embeddings.t())
        score_tensor = torch.clamp(score_tensor, min=-1.0, max=1.0)

        valid_score_mask = left_valid.unsqueeze(1) & right_valid.unsqueeze(0)
        score_tensor = score_tensor * valid_score_mask.to(score_tensor.dtype)

        if device is not None:
            score_tensor = score_tensor.to(device)
            valid_score_mask = valid_score_mask.to(device)
        if return_validity:
            return score_tensor, valid_score_mask
        return score_tensor
