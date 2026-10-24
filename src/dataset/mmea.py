"""Dataset adapter for MMEA-format graph alignment data."""

from .base import BaseGraphPairDataset
from .readers import read_entity_ids, read_triples


class MMEADataset(BaseGraphPairDataset):
    """Load MMEA triples and alignment pairs into the shared dataset interface."""

    def __init__(self, dataset_path, split_path, device):
        """Read root KG resources plus train/dev alignments from a split dir."""
        super().__init__(device=device)
        self.source_entity_ids = read_entity_ids(dataset_path + "/ent_ids_1")
        self.target_entity_ids = read_entity_ids(dataset_path + "/ent_ids_2")
        self.source_triples = read_triples(dataset_path + "/triples_1")
        self.target_triples = read_triples(dataset_path + "/triples_2")

        self.train_pairs = self.read_alignment_pairs(split_path + "/sup_ent_ids")
        self.dev_pairs = self.read_alignment_pairs(split_path + "/dev_ent_ids")

        dev_pairs_by_side = self.dev_pairs.clone().T
        source_dev_ids, target_dev_ids = dev_pairs_by_side
        self.unmatched_source_ids = source_dev_ids.tolist()
        self.unmatched_target_ids = target_dev_ids.tolist()
