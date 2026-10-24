"""Deterministic label splits; graph resources remain transductive."""
import hashlib
import random

import torch


def dmea_split(pairs, ratio, seed):
    if not 0 < ratio < 1:
        raise ValueError("Training ratio must be between zero and one.")
    rows = [tuple(map(int, row)) for row in pairs.tolist()]
    if len(rows) != len(set(rows)):
        raise ValueError("Alignment file contains duplicate pairs.")
    random.Random(seed).shuffle(rows)
    n_train = int(len(rows) * ratio)
    n_valid = (len(rows) - n_train) // 2
    if min(n_train, n_valid) < 1:
        raise ValueError("Each alignment split must be nonempty.")
    shuffled = torch.tensor(rows, dtype=torch.long)
    identity = hashlib.sha256(repr((rows, ratio, seed)).encode()).hexdigest()
    return (shuffled[:n_train], shuffled[n_train:n_train+n_valid],
            shuffled[n_train+n_valid:], identity)


def validation_candidates(train, valid, test):
    # Test endpoints never enter the pseudo pool, including many-to-one links.
    forbidden_left = set(train[:, 0].tolist()) | set(test[:, 0].tolist())
    forbidden_right = set(train[:, 1].tolist()) | set(test[:, 1].tolist())
    eligible = [(a, b) for a, b in valid.tolist()
                if a not in forbidden_left and b not in forbidden_right]
    return (list(dict.fromkeys(a for a, _ in eligible)),
            list(dict.fromkeys(b for _, b in eligible)))
