"""Reusable on-disk caches for dense modality embedding tables."""

import hashlib
import os

import torch


def _torch_load(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_or_create_embedding_cache(source_path, cache_dir, namespace, loader):
    """Parse a source embedding file once, then reuse its dense representation."""
    if not cache_dir:
        return loader(source_path)

    source_path = os.path.abspath(source_path)
    stat = os.stat(source_path)
    identity = "{}|{}|{}|{}".format(
        namespace,
        source_path,
        stat.st_size,
        stat.st_mtime_ns,
    )
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:24]
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, "{}_{}.pt".format(namespace, digest))
    if os.path.isfile(cache_path):
        return _torch_load(cache_path)

    value = loader(source_path)
    temporary_path = cache_path + ".tmp.{}".format(os.getpid())
    torch.save(value, temporary_path)
    os.replace(temporary_path, cache_path)
    return value
