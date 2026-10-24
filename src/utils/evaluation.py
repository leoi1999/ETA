import torch

from .metrics import (
    get_target_ranks,
    hit_metrics_from_ranks,
    rank_metrics_from_ranks,
)


NORMALIZATION_ROUNDS = 30
NORMALIZATION_TEMPERATURE = 0.01


def normalize_similarity(
    similarity_matrix,
    temperature=NORMALIZATION_TEMPERATURE,
    rounds=NORMALIZATION_ROUNDS,
):
    """Apply fixed rounds of row/column normalization in log space."""
    if similarity_matrix.dim() != 2:
        raise ValueError("Similarity matrix must be two-dimensional.")
    if temperature <= 0:
        raise ValueError("Normalization temperature must be positive.")
    if rounds <= 0:
        raise ValueError("Normalization rounds must be positive.")

    normalized_logits = similarity_matrix.clone()
    normalized_logits.div_(temperature)
    for _ in range(rounds):
        normalized_logits.sub_(
            torch.logsumexp(
                normalized_logits,
                dim=1,
                keepdim=True,
            )
        )
        normalized_logits.sub_(
            torch.logsumexp(
                normalized_logits,
                dim=0,
                keepdim=True,
            )
        )
    return normalized_logits.exp_()


def evaluate_normalized_similarity(
    similarity_matrix,
    len_dev,
    return_details=False,
    normalization_temperature=NORMALIZATION_TEMPERATURE,
    normalization_rounds=NORMALIZATION_ROUNDS,
):
    """Normalize a dev similarity matrix and evaluate right-to-left ranking."""
    expected_shape = (len_dev, len_dev)
    if tuple(similarity_matrix.shape) != expected_shape:
        raise ValueError(
            "Similarity matrix shape {} does not match expected dev shape {}.".format(
                tuple(similarity_matrix.shape),
                expected_shape,
            )
        )

    normalized_scores = normalize_similarity(
        similarity_matrix.T,
        temperature=normalization_temperature,
        rounds=normalization_rounds,
    )
    target_indices = torch.arange(len_dev, device=normalized_scores.device)
    ranks = get_target_ranks(
        normalized_scores,
        target_indices,
    )
    result = hit_metrics_from_ranks(ranks, start="\t\t")
    result["MR"], result["MRR"] = rank_metrics_from_ranks(
        ranks,
        start="\t\t",
    )

    if return_details:
        row_indices = target_indices
        top1_scores, top1_indices = torch.max(normalized_scores, dim=1)
        true_scores = normalized_scores[row_indices, row_indices]
        result["prediction_details"] = {
            "row_side": "right",
            "col_side": "left",
            "top1_indices": top1_indices.detach().cpu().tolist(),
            "top1_scores": top1_scores.detach().cpu().tolist(),
            "true_scores": true_scores.detach().cpu().tolist(),
            "true_ranks": ranks.detach().cpu().tolist(),
        }

    return result
