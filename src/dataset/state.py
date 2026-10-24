import time
from dataclasses import dataclass
import math

import torch
from torch.utils.data import DataLoader, Sampler

from .pseudo_pairs import (
    choose_best_pairs,
    find_pseudo_pairs,
    remove_conflicting_pairs,
)


class BalancedBatchSampler(Sampler):
    """Split an epoch into near-equal batches without a tiny final batch."""

    def __init__(self, dataset_size, target_batch_size, shuffle=True):
        if dataset_size < 0:
            raise ValueError("dataset_size must be non-negative")
        if target_batch_size <= 0:
            raise ValueError("target_batch_size must be positive")
        self.dataset_size = int(dataset_size)
        self.target_batch_size = int(target_batch_size)
        self.shuffle = bool(shuffle)
        if self.dataset_size == 0:
            self.batch_count = 0
        else:
            approximate_count = self.dataset_size / float(self.target_batch_size)
            self.batch_count = max(1, int(math.floor(approximate_count + 0.5)))

    @property
    def batch_sizes(self):
        if self.batch_count == 0:
            return []
        base_size, extra = divmod(self.dataset_size, self.batch_count)
        return [
            base_size + int(batch_index < extra)
            for batch_index in range(self.batch_count)
        ]

    def __iter__(self):
        if self.dataset_size == 0:
            return
        indices = (
            torch.randperm(self.dataset_size).tolist()
            if self.shuffle
            else list(range(self.dataset_size))
        )
        offset = 0
        for batch_size in self.batch_sizes:
            yield indices[offset:offset + batch_size]
            offset += batch_size

    def __len__(self):
        return self.batch_count


@dataclass
class DatasetState:
    dataset_name: str
    train_dataset: object
    dev_pairs: object
    unmatched_source_ids: list
    unmatched_target_ids: list
    candidate_source_ids: list
    candidate_target_ids: list
    source_entity_ids: list
    target_entity_ids: list
    entity_feature_edges: object
    relation_feature_edges: object
    entity_count: int
    relation_feature_dim: int
    entity_adjacency_edges: object
    edge_relation_indices: object
    edge_relation_values: object
    edge_count: int
    test_pairs: object = None
    split_identity: str = None

    def __post_init__(self):
        self.supervised_train_pairs = self.train_dataset.pairs.detach().cpu().clone()
        self.allowed_pseudo_source_ids = set(self.candidate_source_ids)
        self.allowed_pseudo_target_ids = set(self.candidate_target_ids)
        self.dev_source_index = {int(v): i for i, v in reversed(list(enumerate(self.dev_pairs[:, 0].tolist())))}
        self.dev_target_index = {int(v): i for i, v in reversed(list(enumerate(self.dev_pairs[:, 1].tolist())))}
        self.excluded_source_indices = set()
        self.excluded_target_indices = set()
        self.source_id_to_candidate_index = {
            entity_id: index
            for index, entity_id in enumerate(self.candidate_source_ids)
        }
        self.target_id_to_candidate_index = {
            entity_id: index
            for index, entity_id in enumerate(self.candidate_target_ids)
        }
        self.unmatched_source_lookup = set(self.unmatched_source_ids)
        self.unmatched_target_lookup = set(self.unmatched_target_ids)

    def log_summary(self, logger):
        logger.info("Training pairs: %s", len(self.train_dataset))
        logger.info("Validation pairs: %s", len(self.dev_pairs))
        if self.test_pairs is not None:
            logger.info("Held-out test pairs: %s | split identity: %s", len(self.test_pairs), self.split_identity)
        logger.info("Pseudo candidate entities: source=%s target=%s", len(self.candidate_source_ids), len(self.candidate_target_ids))
        logger.info(
            "Entity feature edge shape: %s | device: %s",
            self.entity_feature_edges.shape,
            self.entity_feature_edges.device,
        )
        logger.info("Relation feature edge shape: %s", self.relation_feature_edges.shape)
        logger.info("Entity count: %s (E1=%s, E2=%s)", self.entity_count, len(self.source_entity_ids), len(self.target_entity_ids))
        logger.info("Relation feature dimension: %s", self.relation_feature_dim)
        logger.info("Entity adjacency edge shape: %s", self.entity_adjacency_edges.shape)
        logger.info("Edge-relation index shape: %s", self.edge_relation_indices.shape)
        logger.info("Edge-relation value shape: %s", self.edge_relation_values.shape)
        logger.info("Graph edge count: %s", self.edge_count)

    def active_row_indices(self):
        return [
            index
            for index in range(len(self.candidate_source_ids))
            if index not in self.excluded_source_indices
        ]

    def active_col_indices(self):
        return [
            index
            for index in range(len(self.candidate_target_ids))
            if index not in self.excluded_target_indices
        ]

    def train_batch_count(self, batch_size, balanced=True):
        if balanced:
            return len(
                BalancedBatchSampler(
                    dataset_size=len(self.train_dataset),
                    target_batch_size=batch_size,
                    shuffle=False,
                )
            )
        return int(math.ceil(len(self.train_dataset) / float(batch_size)))

    def build_train_loader(self, batch_size, shuffle=True, balanced=True):
        if balanced:
            return DataLoader(
                self.train_dataset,
                batch_sampler=BalancedBatchSampler(
                    dataset_size=len(self.train_dataset),
                    target_batch_size=batch_size,
                    shuffle=shuffle,
                ),
            )
        return DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=False,
        )

    def apply_pseudo_labels(
        self,
        model,
        graph_embeddings,
        threshold,
        threshold_weight,
        conflict_policy,
        logger,
    ):
        started_at = time.time()

        with torch.no_grad():
            # Keep graph rows/columns in the immutable candidate-array order.
            # The active pool is represented by exclusion indices; using the
            # compact unmatched lists here changes ordering after each round
            # and desynchronizes graph scores from modality scores.
            active_row_indices = self.active_row_indices()
            active_col_indices = self.active_col_indices()
            source_candidate_ids = torch.as_tensor(
                [self.candidate_source_ids[i] for i in active_row_indices],
                dtype=torch.long,
            )
            target_candidate_ids = torch.as_tensor(
                [self.candidate_target_ids[i] for i in active_col_indices],
                dtype=torch.long,
            )
            logger.info(
                "Pseudo-label candidate pool sizes: source=%s, target=%s",
                len(source_candidate_ids),
                len(target_candidate_ids),
            )

            if not len(source_candidate_ids) or not len(target_candidate_ids):
                return {'threshold': threshold, 'added_pair_count': 0}
            graph_scores = None
            if graph_embeddings is not None:
                graph_scores = model.compute_graph_similarity(
                    dataset_state=self,
                    source_entity_ids=source_candidate_ids,
                    target_entity_ids=target_candidate_ids,
                )
                if graph_scores is None:
                    left_vec = graph_embeddings[source_candidate_ids]
                    right_vec = graph_embeddings[target_candidate_ids]
                    left_vec = left_vec / (torch.linalg.norm(left_vec, dim=-1, keepdim=True) + 1e-5)
                    right_vec = right_vec / (torch.linalg.norm(right_vec, dim=-1, keepdim=True) + 1e-5)
                    graph_scores = left_vec @ right_vec.T

            logger.info(
                "Pseudo-label exclusion set sizes: source=%s, target=%s",
                len(self.excluded_source_indices),
                len(self.excluded_target_indices),
            )

            scores = model.compute_fused_similarity(
                graph_scores=graph_scores,
                row_indices=[self.dev_source_index[int(e)] for e in source_candidate_ids],
                col_indices=[self.dev_target_index[int(e)] for e in target_candidate_ids],
            )
            self._validate_pseudo_label_pool(active_row_indices, active_col_indices, scores, logger)
            # ``active_*_indices`` are positions in the retained, immutable
            # pseudo candidate arrays.  ``remaining_*_entity_ids`` shrinks
            # after every pseudo-label round, so indexing that compacted list
            # with retained positions fails from the second round onward.
            active_source_ids = [
                self.candidate_source_ids[index]
                for index in active_row_indices
            ]
            active_target_ids = [
                self.candidate_target_ids[index]
                for index in active_col_indices
            ]
            self._validate_active_candidate_order(
                active_source_ids,
                active_target_ids,
            )

            new_pair = find_pseudo_pairs(
                scores,
                active_source_ids,
                active_target_ids,
                threshold,
            )

            if conflict_policy == 1:
                scored_pairs = remove_conflicting_pairs(new_pair)
            else:
                scored_pairs = choose_best_pairs(new_pair)

            max_pairs = int(getattr(getattr(model, 'args', None),
                                    'max_pseudo_pairs_per_round', 0))
            if max_pairs > 0 and len(scored_pairs) > max_pairs:
                # Keep only the highest-confidence mutual top-1 matches.
                scored_pairs = set(sorted(
                    scored_pairs, key=lambda pair: pair[2], reverse=True
                )[:max_pairs])

            new_pair = {(e1, e2) for e1, e2, _ in scored_pairs}

            self._update_training_pairs(new_pair, logger)
            self._update_candidate_sets(new_pair, logger)

            threshold = threshold * threshold_weight
            pool_path = getattr(getattr(model, 'args', None), 'pseudo_pool_path', '')
            if pool_path and new_pair:
                import os
                existing = torch.empty((0, 2), dtype=torch.long)
                records = []
                if os.path.isfile(pool_path):
                    payload = torch.load(pool_path, map_location='cpu', weights_only=True)
                    existing = payload.get('pairs', payload) if isinstance(payload, dict) else payload
                    records = payload.get('records', []) if isinstance(payload, dict) else []
                    if not isinstance(existing, torch.Tensor):
                        existing = torch.empty((0, 2), dtype=torch.long)
                additions = torch.tensor(sorted(new_pair), dtype=torch.long)
                merged = torch.unique(torch.cat((existing.reshape(-1, 2), additions), dim=0), dim=0)
                known = {tuple(row) for row in existing.reshape(-1, 2).tolist()}
                stage = str(getattr(getattr(model, 'args', None), 'modules', 'unknown'))
                epoch = int(getattr(getattr(model, 'args', None), '_pseudo_epoch', -1))
                records.extend({
                    'source_id': int(a), 'target_id': int(b), 'score': float(score),
                    'stage': stage, 'epoch': epoch,
                } for a, b, score in scored_pairs if (a, b) not in known)
                os.makedirs(os.path.dirname(pool_path) or '.', exist_ok=True)
                torch.save({'pairs': merged, 'records': records, 'split_identity': self.split_identity}, pool_path)
            elapsed = time.time() - started_at
            if len(new_pair):
                logger.info(
                    "Average pseudo-pair generation time: %.6f seconds/pair",
                    elapsed / len(new_pair),
                )

        return {
            'threshold': threshold,
            'added_pair_count': len(new_pair),
        }

    def _validate_pseudo_label_pool(self, active_row_indices, active_col_indices, scores, logger):
        if scores.shape[1] != len(active_col_indices):
            raise RuntimeError("Pseudo-label score matrix column count mismatch.")
        if scores.shape[0] != len(active_row_indices):
            logger.error(
                "Score matrix row count mismatch: score_rows=%s, expected_rows=%s",
                scores.shape[0],
                len(active_row_indices),
            )
            raise RuntimeError("Pseudo-label score matrix row count mismatch.")

    def _validate_active_candidate_order(
        self,
        active_source_ids,
        active_target_ids,
    ):
        """Ensure graph-score rows still match fixed modality-score indices."""
        if set(active_source_ids) != set(self.unmatched_source_ids):
            raise RuntimeError(
                "Source pseudo-label candidate order is inconsistent with the "
                "remaining graph candidate pool."
            )
        if set(active_target_ids) != set(self.unmatched_target_ids):
            raise RuntimeError(
                "Target pseudo-label candidate order is inconsistent with the "
                "remaining graph candidate pool."
            )

    def _update_training_pairs(self, new_pair, logger):
        if not new_pair:
            logger.info(
                "Pseudo-label update applied: train_pair_shape=%s, added_pairs=0",
                self.train_dataset.pairs.shape,
            )
            return

        if self.test_pairs is not None:
            if any(a not in self.allowed_pseudo_source_ids or b not in self.allowed_pseudo_target_ids for a, b in new_pair):
                raise ValueError("Pseudo labels must use validation-only candidates; test/train entities are forbidden.")
        train_pairs = self.train_dataset.pairs
        new_pair_tensor = torch.tensor(
            list(new_pair),
            dtype=train_pairs.dtype,
            device=train_pairs.device,
        )
        self.train_dataset.append_pairs(new_pair_tensor.cpu())
        logger.info(
            "Pseudo-label update applied: train_pair_shape=%s, added_pairs=%s",
            self.train_dataset.pairs.shape,
            len(new_pair),
        )

    def _update_candidate_sets(self, new_pair, logger):
        count = 0
        for e1, e2 in new_pair:
            if (
                e1 not in self.unmatched_source_lookup
                or e2 not in self.unmatched_target_lookup
            ):
                continue

            source_index = self.source_id_to_candidate_index.get(e1)
            target_index = self.target_id_to_candidate_index.get(e2)
            if source_index is None or target_index is None:
                logger.warning(
                    "Pseudo pair (%s, %s) was skipped because it is absent from the retained candidate pools.",
                    e1,
                    e2,
                )
                continue

            if (
                source_index in self.excluded_source_indices
                or target_index in self.excluded_target_indices
            ):
                logger.warning(
                    "Pseudo pair indices already excluded: left_index=%s excluded=%s | right_index=%s excluded=%s",
                    source_index,
                    source_index in self.excluded_source_indices,
                    target_index,
                    target_index in self.excluded_target_indices,
                )
                continue

            self.excluded_source_indices.add(source_index)
            self.excluded_target_indices.add(target_index)
            count += 1

            if e1 in self.unmatched_source_lookup:
                self.unmatched_source_lookup.remove(e1)
                try:
                    self.unmatched_source_ids.remove(e1)
                except ValueError:
                    logger.warning(
                        "Pseudo pair left entity %s could not be removed from the remaining candidate set.",
                        e1,
                    )

            if e2 in self.unmatched_target_lookup:
                self.unmatched_target_lookup.remove(e2)
                try:
                    self.unmatched_target_ids.remove(e2)
                except ValueError:
                    logger.warning(
                        "Pseudo pair right entity %s could not be removed from the remaining candidate set.",
                        e2,
                    )

        logger.info(
            "Pseudo-label summary: proposed_pairs=%s, successfully_removed_pairs=%s",
            len(new_pair),
            count,
        )
