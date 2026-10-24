import os
import random

import numpy as np
import torch
from torch import Tensor


@torch.no_grad()
def get_target_ranks(
    sim_matrix: Tensor,
    target_indices: Tensor,
    batch_size=2048,
):
    """Compute exact target ranks without sorting rows that have no ties.

    For continuous scores, rank is one plus the number of candidates with a
    strictly larger score. If a target score is tied, use ``argsort`` for only
    those rows so deterministic tie ordering is preserved.
    """
    if sim_matrix.dim() != 2:
        raise ValueError("Similarity matrix must be two-dimensional.")
    if target_indices.dim() != 1 or target_indices.numel() != sim_matrix.shape[0]:
        raise ValueError(
            "target_indices must contain one target column per similarity row."
        )
    if batch_size <= 0:
        raise ValueError("Rank batch size must be positive.")

    target_indices = target_indices.to(device=sim_matrix.device, dtype=torch.long)
    rank_parts = []
    for begin in range(0, sim_matrix.shape[0], batch_size):
        end = min(begin + batch_size, sim_matrix.shape[0])
        score_rows = sim_matrix[begin:end]
        batch_targets = target_indices[begin:end]
        row_indices = torch.arange(
            end - begin,
            device=sim_matrix.device,
        )
        target_scores = score_rows[row_indices, batch_targets]
        ranks = torch.count_nonzero(
            score_rows > target_scores.unsqueeze(1),
            dim=1,
        ) + 1

        tied_rows = torch.count_nonzero(
            score_rows == target_scores.unsqueeze(1),
            dim=1,
        ) > 1
        if torch.any(tied_rows):
            tied_scores = score_rows[tied_rows]
            tied_targets = batch_targets[tied_rows]
            sorted_indices = torch.argsort(
                tied_scores,
                dim=1,
                descending=True,
            )
            deterministic_tied_ranks = torch.argmax(
                (sorted_indices == tied_targets.unsqueeze(1)).to(torch.long),
                dim=1,
            ) + 1
            ranks[tied_rows] = deterministic_tied_ranks
        rank_parts.append(ranks)

    if not rank_parts:
        return torch.empty(0, dtype=torch.long, device=sim_matrix.device)
    return torch.cat(rank_parts, dim=0)


def hit_metrics_from_ranks(ranks: Tensor, k_list=(1, 3, 5, 10), start="\t"):
    total = int(ranks.numel())
    print(start + "total is ", total)
    result = {}
    for k in k_list:
        hit_value = float(torch.count_nonzero(ranks <= k)) / float(total)
        result["hits@{}".format(k)] = hit_value
        print("{2}hits@{0} is {1}".format(k, hit_value, start))
    return result


def rank_metrics_from_ranks(ranks: Tensor, start="\t"):
    ranks = ranks.to(float)
    mr = ranks.mean()
    mrr = (1.0 / ranks).mean()
    print("{0}MR is {1}".format(start, mr))
    print("{0}MRR is {1}".format(start, mrr))
    return mr.item(), mrr.item()


def seed_torch(seed=1029):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
