"""Shared dataset utilities for graph-based entity alignment.

This base class provides two groups of functionality:
1. loading alignment pairs into a `Dataset`-compatible container,
2. turning KG triples into the tensor layout expected by the graph encoder.

Subclasses such as `MMEADataset` are responsible for deciding which files to
read, while this class handles the generic graph reconstruction logic.
"""

import torch
from torch.utils.data import Dataset


class BaseGraphPairDataset(Dataset):
    """Base dataset for paired-entity supervision plus graph tensors."""

    def __init__(self, device):
        self.device = device
        self.train_pairs = torch.empty((0, 2), dtype=torch.long)

    def __len__(self):
        return len(self.train_pairs)

    def __getitem__(self, index):
        return self.train_pairs[index]

    @staticmethod
    def read_alignment_pairs(file_name):
        """Read tab-separated entity alignment pairs from disk.

        Each valid line should contain two integer entity ids:
        `entity_id_in_source_kg <tab> entity_id_in_target_kg`.
        Malformed lines are skipped so slightly noisy files do not stop loading.
        """
        alignment_pairs = []
        with open(file_name, 'r', encoding='utf-8') as file:
            for line in file:
                try:
                    e1, e2 = line.split()
                    alignment_pairs.append((int(e1), int(e2)))
                except Exception:
                    continue
        return torch.as_tensor(alignment_pairs, dtype=torch.long)

    def build_graph_tensors(
        self,
        source_triples,
        target_triples,
    ):
        """Merge two KGs and build the sparse tensors used by the model."""
        entity = set()
        all_triples = source_triples + target_triples

        # Reserve relation id 0, then shift every observed relation by +1.
        # This mirrors the original project convention when building the graph.
        rel = {0}
        triples = []
        for head, r, tail in all_triples:
            entity.add(int(head))
            entity.add(int(tail))
            r = int(r)
            rel.add(r + 1)
            triples.append([int(head), r + 1, int(tail)])
        entity_count = max(entity) + 1 if entity else 0
        relation_count = max(rel) + 1 if rel else 0

        return self.build_sparse_graph(
            triples=triples,
            entity_count=entity_count,
            relation_count=relation_count,
        )

    def build_sparse_graph(self, triples, entity_count, relation_count):
        """Build graph tensors directly without dense adjacency matrices."""
        adjacency_edges = set()
        entity_feature_edges = {(entity_id, entity_id) for entity_id in range(entity_count)}
        relation_feature_edges = set()
        edge_relation_rows = []

        for head, relation_id, tail in triples:
            adjacency_edges.add((head, tail))
            adjacency_edges.add((tail, head))
            entity_feature_edges.add((head, tail))
            entity_feature_edges.add((tail, head))

            relation_feature_edges.add((tail, relation_id))
            relation_feature_edges.add((head, relation_count + relation_id))

            edge_relation_rows.append((head, tail, relation_id))
            edge_relation_rows.append(
                (tail, head, relation_count + relation_id)
            )

        sorted_adjacency_edges = sorted(adjacency_edges)
        edge_index_by_pair = {
            edge_pair: edge_index
            for edge_index, edge_pair in enumerate(sorted_adjacency_edges)
        }
        relation_ids_by_edge = [[] for _ in sorted_adjacency_edges]
        for head, tail, relation_id in edge_relation_rows:
            relation_ids_by_edge[edge_index_by_pair[(head, tail)]].append(relation_id)

        edge_relation_indices = []
        edge_relation_values = []
        for edge_index, relation_ids in enumerate(relation_ids_by_edge):
            relation_weight = 1.0 / float(len(relation_ids))
            for relation_id in relation_ids:
                edge_relation_indices.append((edge_index, relation_id))
                edge_relation_values.append(relation_weight)

        def edge_tensor(edge_pairs):
            if not edge_pairs:
                return torch.empty((2, 0), dtype=torch.long, device=self.device)
            return torch.as_tensor(
                edge_pairs,
                dtype=torch.long,
                device=self.device,
            ).T.contiguous()

        entity_adjacency_edges = edge_tensor(sorted_adjacency_edges)
        entity_feature_edge_tensor = edge_tensor(sorted(entity_feature_edges))
        relation_feature_edge_tensor = edge_tensor(sorted(relation_feature_edges))
        edge_relation_index_tensor = edge_tensor(edge_relation_indices)
        edge_relation_value_tensor = torch.as_tensor(
            edge_relation_values,
            dtype=torch.float32,
            device=self.device,
        )

        return (
            entity_feature_edge_tensor,
            relation_feature_edge_tensor,
            entity_count,
            relation_count * 2,
            entity_adjacency_edges,
            edge_relation_index_tensor,
            edge_relation_value_tensor,
            len(sorted_adjacency_edges),
        )
