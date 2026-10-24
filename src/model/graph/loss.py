import torch
import torch.nn as nn


class BidirectionalGraphAlignmentLoss(nn.Module):
    def __init__(
        self,
        entity_count,
        source_entity_ids,
        target_entity_ids,
        beta=30,
    ):
        super().__init__()
        self.beta = float(beta)
        if self.beta <= 0:
            raise ValueError("hard-negative beta must be positive")
        self.entity_count = int(entity_count)

        source_entity_ids = torch.as_tensor(source_entity_ids, dtype=torch.long)
        target_entity_ids = torch.as_tensor(target_entity_ids, dtype=torch.long)
        if source_entity_ids.ndim != 1 or target_entity_ids.ndim != 1:
            raise ValueError("source and target entity ids must be one-dimensional")
        if source_entity_ids.numel() == 0 or target_entity_ids.numel() == 0:
            raise ValueError("source and target candidate sets must be non-empty")
        all_candidate_ids = torch.cat([source_entity_ids, target_entity_ids])
        if torch.any(all_candidate_ids < 0) or torch.any(
            all_candidate_ids >= self.entity_count
        ):
            raise ValueError("candidate entity id is outside the embedding table")
        if torch.unique(all_candidate_ids).numel() != all_candidate_ids.numel():
            raise ValueError("source and target entity candidate sets must be disjoint")

        source_candidate_index = torch.full(
            (self.entity_count,),
            -1,
            dtype=torch.long,
        )
        target_candidate_index = torch.full(
            (self.entity_count,),
            -1,
            dtype=torch.long,
        )
        source_candidate_index[source_entity_ids] = torch.arange(
            source_entity_ids.numel(), dtype=torch.long
        )
        target_candidate_index[target_entity_ids] = torch.arange(
            target_entity_ids.numel(), dtype=torch.long
        )

        self.register_buffer(
            "source_entity_ids",
            source_entity_ids,
            persistent=False,
        )
        self.register_buffer(
            "target_entity_ids",
            target_entity_ids,
            persistent=False,
        )
        self.register_buffer(
            "source_candidate_index",
            source_candidate_index,
            persistent=False,
        )
        self.register_buffer(
            "target_candidate_index",
            target_candidate_index,
            persistent=False,
        )

    def align_loss(self, pairs, emb):
        def squared_dist(A, B):
            row_norms_A = torch.sum(torch.square(A), dim=1)
            row_norms_A = torch.reshape(row_norms_A, [-1, 1])
            row_norms_B = torch.sum(torch.square(B), dim=1)
            row_norms_B = torch.reshape(row_norms_B, [1, -1])
            return row_norms_A + row_norms_B - 2 * torch.matmul(A, B.t())

        if emb.ndim != 2 or emb.shape[0] != self.entity_count:
            raise ValueError(
                "feature must have shape [{}, embedding_dim]".format(
                    self.entity_count
                )
            )

        left_entity_ids = pairs[:, 0].long()
        right_entity_ids = pairs[:, 1].long()
        left_positive_columns = self.target_candidate_index[right_entity_ids]
        right_positive_columns = self.source_candidate_index[left_entity_ids]
        if torch.any(left_positive_columns < 0):
            raise ValueError("a right-side training entity is absent from E2")
        if torch.any(right_positive_columns < 0):
            raise ValueError("a left-side training entity is absent from E1")

        left_embeddings = emb[left_entity_ids]
        right_embeddings = emb[right_entity_ids]
        positive_distances = torch.sum(
            torch.square(left_embeddings - right_embeddings),
            dim=-1,
            keepdim=True,
        )

        # Equation (7): u is compared only with E2, while v is compared only
        # with E1. The embedding table itself contains E1 and E2 jointly.
        left_negative_distances = squared_dist(
            left_embeddings,
            emb[self.target_entity_ids],
        )
        right_negative_distances = squared_dist(
            right_embeddings,
            emb[self.source_entity_ids],
        )
        del left_embeddings, right_embeddings

        def set_positive_candidate_zero(loss_matrix, positive_columns):
            row_indices = torch.arange(
                loss_matrix.shape[0],
                device=loss_matrix.device,
            )
            # For the aligned candidate d == x, Eq. (8) gives
            # x - d == 0 before normalization.
            loss_matrix[row_indices, positive_columns] = 0.0
            return loss_matrix

        left_loss = set_positive_candidate_zero(
            positive_distances - left_negative_distances,
            left_positive_columns,
        )
        right_loss = set_positive_candidate_zero(
            positive_distances - right_negative_distances,
            right_positive_columns,
        )

        del right_negative_distances, left_negative_distances

        right_std = torch.std(
            right_loss,
            dim=-1,
            unbiased=False,
            keepdim=True,
        ).detach().clamp_min(1e-12)
        left_std = torch.std(
            left_loss,
            dim=-1,
            unbiased=False,
            keepdim=True,
        ).detach().clamp_min(1e-12)
        right_loss = (
            right_loss - torch.mean(right_loss, dim=-1, keepdim=True).detach()
        ) / right_std
        left_loss = (
            left_loss - torch.mean(left_loss, dim=-1, keepdim=True).detach()
        ) / left_std

        # Equation (8) uses beta to emphasize difficult candidates.  The
        # additive offset b is intentionally kept fixed at 10, matching the
        # original implementation.
        offset = 10.0
        left_loss = torch.logsumexp(self.beta * left_loss + offset, dim=-1)
        right_loss = torch.logsumexp(self.beta * right_loss + offset, dim=-1)
        # Equation (7) sums both directions and divides only by |H_train|.
        return torch.mean(left_loss + right_loss)

    def forward(self, train_pairs: torch.Tensor, feature):
        if train_pairs.ndim != 2 or train_pairs.shape[1] != 2:
            raise ValueError(
                "train_pairs must contain exactly two columns: left and right entity ids"
            )
        return self.align_loss(train_pairs, feature)
