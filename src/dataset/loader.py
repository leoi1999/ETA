"""Dataset loading helpers used by the training entrypoint.

This module translates parsed runtime arguments into a concrete
`DatasetState`, which is the object consumed later by training and evaluation.
The loading flow is:
1. resolve the dataset name from the user-facing dataset key,
2. build the on-disk dataset path,
3. load graph/pair resources through `MMEADataset`,
4. convert them into the state expected by `src/run.py`.
"""

import os
import logging

import torch

from .pairs import EntityAlignmentDataset
from .mmea import MMEADataset
from .state import DatasetState
from .splits import dmea_split, validation_candidates


DATASET_CACHE_VERSION = 2


def _source_signature(paths):
    return {
        os.path.abspath(path): (os.path.getsize(path), os.stat(path).st_mtime_ns)
        for path in paths
    }


def _dataset_cache_path(args, dataset_name):
    cache_dir = getattr(args, 'dataset_cache_dir', None)
    if not cache_dir:
        return None
    os.makedirs(cache_dir, exist_ok=True)
    ratio = str(args.ratio).replace('.', 'p')
    return os.path.join(
        cache_dir,
        '{}_r{}_{}_s{}_v{}.pt'.format(dataset_name, ratio, args.alignment_split, args.split_seed, DATASET_CACHE_VERSION),
    )


def _state_payload(state, signature):
    return {
        'version': DATASET_CACHE_VERSION,
        'signature': signature,
        'dataset_name': state.dataset_name,
        'train_pairs': state.train_dataset.pairs.detach().cpu(),
        'dev_pairs': state.dev_pairs.detach().cpu(),
        'test_pairs': state.test_pairs,
        'split_identity': state.split_identity,
        'unmatched_source_ids': list(state.unmatched_source_ids),
        'unmatched_target_ids': list(state.unmatched_target_ids),
        'candidate_source_ids': list(state.candidate_source_ids),
        'candidate_target_ids': list(state.candidate_target_ids),
        'source_entity_ids': list(state.source_entity_ids),
        'target_entity_ids': list(state.target_entity_ids),
        'entity_feature_edges': state.entity_feature_edges.detach().cpu(),
        'relation_feature_edges': state.relation_feature_edges.detach().cpu(),
        'entity_count': state.entity_count,
        'relation_feature_dim': state.relation_feature_dim,
        'entity_adjacency_edges': state.entity_adjacency_edges.detach().cpu(),
        'edge_relation_indices': state.edge_relation_indices.detach().cpu(),
        'edge_relation_values': state.edge_relation_values.detach().cpu(),
        'edge_count': state.edge_count,
    }


def _state_from_payload(payload, device):
    return DatasetState(
        dataset_name=payload['dataset_name'],
        dev_pairs=payload['dev_pairs'],
        test_pairs=payload['test_pairs'],
        split_identity=payload['split_identity'],
        train_dataset=EntityAlignmentDataset(payload['train_pairs']),
        unmatched_source_ids=list(payload['unmatched_source_ids']),
        unmatched_target_ids=list(payload['unmatched_target_ids']),
        candidate_source_ids=list(payload['candidate_source_ids']),
        candidate_target_ids=list(payload['candidate_target_ids']),
        source_entity_ids=list(payload['source_entity_ids']),
        target_entity_ids=list(payload['target_entity_ids']),
        entity_feature_edges=payload['entity_feature_edges'].to(device),
        relation_feature_edges=payload['relation_feature_edges'].to(device),
        entity_count=int(payload['entity_count']),
        relation_feature_dim=int(payload['relation_feature_dim']),
        entity_adjacency_edges=payload['entity_adjacency_edges'].to(device),
        edge_relation_indices=payload['edge_relation_indices'].to(device),
        edge_relation_values=payload['edge_relation_values'].to(device),
        edge_count=int(payload['edge_count']),
    )


def _torch_load(path, **kwargs):
    try:
        return torch.load(path, weights_only=False, **kwargs)
    except TypeError:
        return torch.load(path, **kwargs)


def resolve_dataset_name(dataset_key):
    """Map the short dataset key from args to the concrete directory name."""
    dataset_names = {
        'FB_DB': 'FB15K_DB15K',
        'FB_YAGO': 'FB15K_YAGO15K',
    }
    try:
        return dataset_names[dataset_key]
    except KeyError as error:
        raise ValueError('Unsupported dataset key: {}'.format(dataset_key)) from error



def load_dataset_state(args):
    """Load all dataset resources needed for one run.

    The returned `DatasetState` packages both supervision data
    (train/dev pairs) and graph tensors so the rest of the pipeline
    does not need to know how they were read from disk.
    """
    protocol = getattr(args, 'alignment_split', 'auto')
    if getattr(args, 'semi_prediction', False) and protocol == 'legacy':
        raise ValueError("Pseudo-label experiments require --alignment-split dmea (or auto).")
    args.alignment_split = ('dmea' if getattr(args, 'semi_prediction', False) else 'legacy') if protocol == 'auto' else protocol
    args.split_seed = getattr(args, 'split_seed', 43)
    dataset_name = resolve_dataset_name(
        args.dataset,
    )
    # Current MMEA layout:
    #   <data_root>/<dataset_name>/
    #       ent_ids_1, ent_ids_2, triples_1, triples_2, training_attrs_1/2
    #       seed<ratio>/sup_ent_ids, dev_ent_ids, ref_ent_ids
    mmea_dataset_dir = os.path.join(args.data_root, dataset_name)
    split_dir = os.path.join(mmea_dataset_dir, 'seed{}'.format(str(args.ratio)))
    source_paths = [
        os.path.join(mmea_dataset_dir, name)
        for name in ('ent_ids_1', 'ent_ids_2', 'triples_1', 'triples_2')
    ] + [
        os.path.join(split_dir, name)
        for name in ('sup_ent_ids', 'dev_ent_ids')
    ]
    if args.alignment_split == 'dmea':
        source_paths.append(os.path.join(split_dir, 'ref_ent_ids'))
    signature = _source_signature(source_paths)
    cache_path = _dataset_cache_path(args, dataset_name)
    if cache_path and os.path.isfile(cache_path):
        payload = _torch_load(cache_path, map_location='cpu')
        if (
            payload.get('version') == DATASET_CACHE_VERSION
            and payload.get('signature') == signature
        ):
            args._dataset_cache_hit = True
            return _restore_pseudo_pool(_state_from_payload(payload, args.device), args)

    graph_dataset = MMEADataset(
        dataset_path=mmea_dataset_dir,
        split_path=split_dir,
        device=args.device,
    )

    train_pairs, dev_pairs = graph_dataset.train_pairs, graph_dataset.dev_pairs
    test_pairs, split_identity = None, None
    if args.alignment_split == 'dmea':
        all_pairs = graph_dataset.read_alignment_pairs(os.path.join(split_dir, 'ref_ent_ids'))
        train_pairs, dev_pairs, test_pairs, split_identity = dmea_split(all_pairs, args.ratio, args.split_seed)
    train_dataset = EntityAlignmentDataset(train_pairs)

    # These "rest sets" track entities that are not yet matched by the current
    # supervision set and are reused by pseudo-label generation.
    unmatched_source_ids = graph_dataset.unmatched_source_ids
    unmatched_target_ids = graph_dataset.unmatched_target_ids
    if test_pairs is not None:
        unmatched_source_ids, unmatched_target_ids = validation_candidates(train_pairs, dev_pairs, test_pairs)
    candidate_source_ids = unmatched_source_ids.copy()
    candidate_target_ids = unmatched_target_ids.copy()

    # Rebuild the graph-side sparse structures used by the model. The runtime
    # state stores them once so later stages can share the same tensors.
    (
        entity_feature_edges,
        relation_feature_edges,
        entity_count,
        relation_feature_dim,
        entity_adjacency_edges,
        edge_relation_indices,
        edge_relation_values,
        edge_count,
    ) = graph_dataset.build_graph_tensors(
        graph_dataset.source_triples,
        graph_dataset.target_triples,
    )

    state = DatasetState(
        dataset_name=dataset_name,
        dev_pairs=dev_pairs,
        test_pairs=test_pairs,
        split_identity=split_identity,
        train_dataset=train_dataset,
        unmatched_source_ids=unmatched_source_ids,
        unmatched_target_ids=unmatched_target_ids,
        candidate_source_ids=candidate_source_ids,
        candidate_target_ids=candidate_target_ids,
        source_entity_ids=graph_dataset.source_entity_ids,
        target_entity_ids=graph_dataset.target_entity_ids,
        entity_feature_edges=entity_feature_edges,
        relation_feature_edges=relation_feature_edges,
        entity_count=entity_count,
        relation_feature_dim=relation_feature_dim,
        entity_adjacency_edges=entity_adjacency_edges,
        edge_relation_indices=edge_relation_indices,
        edge_relation_values=edge_relation_values,
        edge_count=edge_count,
    )
    args._dataset_cache_hit = False
    if cache_path:
        temporary_path = cache_path + '.tmp.{}'.format(os.getpid())
        torch.save(_state_payload(state, signature), temporary_path)
        os.replace(temporary_path, cache_path)
    return _restore_pseudo_pool(state, args)


def _restore_pseudo_pool(state, args):
    # Cache only pristine supervision; restore shared labels on every load.
    pool_path = getattr(args, 'pseudo_pool_path', '')
    if pool_path and os.path.isfile(pool_path):
        payload = _torch_load(pool_path, map_location='cpu')
        if state.split_identity and (not isinstance(payload, dict) or payload.get('split_identity') != state.split_identity):
            raise ValueError("Shared pseudo pool belongs to a different or legacy alignment split.")
        pairs = payload.get('pairs', payload) if isinstance(payload, dict) else payload
        if isinstance(pairs, torch.Tensor) and pairs.numel():
            restored = {tuple(map(int, row)) for row in pairs.reshape(-1, 2).tolist()}
            state._update_training_pairs(restored, logging.getLogger(__name__))
            state._update_candidate_sets(restored, logging.getLogger(__name__))
    return state
