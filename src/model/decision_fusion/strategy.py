from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class BaseDecisionFusion(nn.Module, ABC):
    SCORE_ORDER = ("graph", "attribute", "visual")

    def __init__(self):
        super().__init__()

    @abstractmethod
    def fuse(
        self,
        graph_scores=None,
        modality_score_tensors=None,
        modality_validity_masks=None,
        return_weights=False,
    ):
        raise NotImplementedError

    def get_weight_summary(self):
        return None

    def start_weight_summary_accumulation(self):
        return None

    def finish_weight_summary_accumulation(self):
        return self.get_weight_summary()


class DynamicDecisionFusion(BaseDecisionFusion):
    """Pair-specific Dynamic Decision Fusion over graph, visual, and attribute scores."""

    def __init__(
        self,
        hidden_dim=32,
        temperature=1.0,
        chunk_size=512,
        initial_weights=None,
    ):
        super().__init__()
        if hidden_dim <= 0:
            raise ValueError("fusion hidden dimension must be positive")
        if temperature <= 0:
            raise ValueError("Dynamic Decision Fusion temperature must be positive")
        if chunk_size <= 0:
            raise ValueError("fusion chunk size must be positive")

        self.hidden_dim = int(hidden_dim)
        self.temperature = float(temperature)
        self.chunk_size = int(chunk_size)
        self.initial_weights = self._normalize_initial_weights(initial_weights)
        modality_count = len(self.SCORE_ORDER)
        self.gating_network = nn.Sequential(
            nn.Linear(modality_count, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, modality_count),
        )
        # Start from exact uniform fusion and learn pair-specific deviations.
        nn.init.zeros_(self.gating_network[-1].weight)
        nn.init.zeros_(self.gating_network[-1].bias)
        if self.initial_weights is not None:
            initial_logits = self.temperature * torch.log(
                torch.as_tensor(
                    self.initial_weights,
                    dtype=self.gating_network[-1].bias.dtype,
                )
            )
            with torch.no_grad():
                self.gating_network[-1].bias.copy_(initial_logits)
        self._last_weight_summary = None
        self._weight_summary_accumulator = None

    @classmethod
    def _normalize_initial_weights(cls, initial_weights):
        if initial_weights is None:
            return None
        if len(initial_weights) != len(cls.SCORE_ORDER):
            raise ValueError(
                "fusion initial weights must follow graph, attribute, visual order"
            )
        weights = tuple(float(value) for value in initial_weights)
        if any(value <= 0 for value in weights):
            raise ValueError("fusion initial weights must be strictly positive")
        weight_sum = sum(weights)
        if not torch.isfinite(torch.as_tensor(weight_sum)) or weight_sum <= 0:
            raise ValueError("fusion initial weights must have a finite positive sum")
        return tuple(value / weight_sum for value in weights)

    def start_weight_summary_accumulation(self):
        device = self._fusion_device()
        self._weight_summary_accumulator = {
            "sum": torch.zeros(len(self.SCORE_ORDER), dtype=torch.float64, device=device),
            "square_sum": torch.zeros(
                len(self.SCORE_ORDER),
                dtype=torch.float64,
                device=device,
            ),
            "count": torch.zeros(
                len(self.SCORE_ORDER),
                dtype=torch.float64,
                device=device,
            ),
        }
        self._last_weight_summary = None

    def _set_weight_summary(self, weight_sum, weight_square_sum, weight_count):
        safe_count = weight_count.clamp_min(1.0)
        weight_mean = weight_sum / safe_count
        weight_variance = (weight_square_sum / safe_count) - weight_mean.square()
        weight_std = torch.sqrt(weight_variance.clamp_min(0.0))
        self._last_weight_summary = {
            module_name: {
                "mean": float(weight_mean[index].cpu()),
                "std": float(weight_std[index].cpu()),
                "count": int(weight_count[index].cpu()),
            }
            for index, module_name in enumerate(self.SCORE_ORDER)
            if weight_count[index] > 0
        }

    def finish_weight_summary_accumulation(self):
        accumulator = self._weight_summary_accumulator
        if accumulator is not None:
            self._set_weight_summary(
                accumulator["sum"],
                accumulator["square_sum"],
                accumulator["count"],
            )
        self._weight_summary_accumulator = None
        return self._last_weight_summary

    @staticmethod
    def _cosine_score(score):
        """Keep gate inputs as the raw cosine scores defined in Eq. (21)."""
        return score.clamp(min=-1.0, max=1.0)

    def _collect_scores(self, graph_scores, modality_score_tensors):
        score_by_name = {}
        if graph_scores is not None:
            score_by_name["graph"] = graph_scores
        if modality_score_tensors:
            score_by_name.update(modality_score_tensors)
        if not score_by_name:
            raise ValueError("No active modules selected for similarity computation.")

        reference = next(iter(score_by_name.values()))
        if reference.ndim != 2:
            raise ValueError("Fusion score tensors must be two-dimensional matrices.")
        for module_name, score in score_by_name.items():
            if module_name not in self.SCORE_ORDER:
                raise ValueError("Unsupported fusion score module: {}".format(module_name))
            if score.shape != reference.shape:
                raise ValueError(
                    "Fusion score shape mismatch for {}: {} != {}".format(
                        module_name,
                        tuple(score.shape),
                        tuple(reference.shape),
                    )
                )
            if score.device != reference.device:
                score_by_name[module_name] = score.to(reference.device)
        return score_by_name, reference

    def _build_chunk(
        self,
        score_by_name,
        modality_validity_masks,
        start,
        end,
        reference,
    ):
        score_parts = []
        validity_parts = []
        masks = modality_validity_masks or {}
        for module_name in self.SCORE_ORDER:
            score = score_by_name.get(module_name)
            if score is None:
                score_parts.append(torch.zeros_like(reference[start:end]))
                validity_parts.append(
                    torch.zeros_like(reference[start:end], dtype=torch.bool)
                )
                continue

            cosine_score = self._cosine_score(score[start:end])
            score_parts.append(cosine_score)
            validity = masks.get(module_name)
            if validity is None:
                validity = torch.ones_like(cosine_score, dtype=torch.bool)
            else:
                if validity.shape != reference.shape:
                    raise ValueError(
                        "Fusion validity-mask shape mismatch for {}: {} != {}".format(
                            module_name,
                            tuple(validity.shape),
                            tuple(reference.shape),
                        )
                    )
                validity = validity[start:end].to(
                    device=reference.device,
                    dtype=torch.bool,
                )
            validity_parts.append(validity)

        score_features = torch.stack(score_parts, dim=-1)
        validity_mask = torch.stack(validity_parts, dim=-1)
        no_valid_evidence = ~torch.any(validity_mask, dim=-1)
        if torch.any(no_valid_evidence):
            # Preserve a finite zero score for pairs where every available
            # modality reports missing evidence (for example image-only runs).
            present_mask = torch.as_tensor(
                [name in score_by_name for name in self.SCORE_ORDER],
                dtype=torch.bool,
                device=reference.device,
            )
            validity_mask[no_valid_evidence] = present_mask
        return score_features, validity_mask

    def compute_weights(self, score_features, validity_mask):
        logits = self.gating_network(score_features) / self.temperature
        logits = logits.masked_fill(~validity_mask, torch.finfo(logits.dtype).min)
        return torch.softmax(logits, dim=-1)

    def _fusion_device(self):
        return next(self.gating_network.parameters()).device

    def fuse(
        self,
        graph_scores=None,
        modality_score_tensors=None,
        modality_validity_masks=None,
        return_weights=False,
    ):
        score_by_name, reference = self._collect_scores(
            graph_scores,
            modality_score_tensors,
        )
        if len(score_by_name) == 1:
            module_name, only_score = next(iter(score_by_name.items()))
            validity = (modality_validity_masks or {}).get(module_name)
            if validity is None:
                validity = torch.ones_like(only_score, dtype=torch.bool)
            else:
                validity = validity.to(device=only_score.device, dtype=torch.bool)
            fused_score = only_score.clone()
            valid_count = validity.sum().double()
            accumulator = self._weight_summary_accumulator
            if accumulator is None:
                self._last_weight_summary = {
                    module_name: {
                        "mean": 1.0,
                        "std": 0.0,
                        "count": int(valid_count.cpu()),
                    }
                }
            else:
                module_index = self.SCORE_ORDER.index(module_name)
                accumulator["sum"][module_index] += valid_count
                accumulator["square_sum"][module_index] += valid_count
                accumulator["count"][module_index] += valid_count
            if not return_weights:
                return fused_score
            weights = torch.zeros(
                (*only_score.shape, len(self.SCORE_ORDER)),
                dtype=only_score.dtype,
                device=only_score.device,
            )
            module_index = self.SCORE_ORDER.index(module_name)
            weights[..., module_index] = 1.0
            return fused_score, weights

        fused_score = torch.empty_like(reference)
        all_weights = [] if return_weights else None
        fusion_device = self._fusion_device()
        weight_sum = torch.zeros(
            len(self.SCORE_ORDER), dtype=torch.float64, device=fusion_device
        )
        weight_square_sum = torch.zeros_like(weight_sum)
        weight_count = torch.zeros_like(weight_sum)

        for start in range(0, reference.shape[0], self.chunk_size):
            end = min(start + self.chunk_size, reference.shape[0])
            score_features, validity_mask = self._build_chunk(
                score_by_name=score_by_name,
                modality_validity_masks=modality_validity_masks,
                start=start,
                end=end,
                reference=reference,
            )
            if score_features.device != fusion_device:
                score_features = score_features.to(fusion_device)
                validity_mask = validity_mask.to(fusion_device)
            weights = self.compute_weights(score_features, validity_mask)
            fused_chunk = torch.sum(weights * score_features, dim=-1)
            fused_score[start:end] = fused_chunk.to(reference.device)

            detached_weights = weights.detach().double()
            detached_validity = validity_mask.detach().double()
            weight_sum += torch.sum(detached_weights, dim=(0, 1))
            weight_square_sum += torch.sum(detached_weights.square(), dim=(0, 1))
            weight_count += torch.sum(detached_validity, dim=(0, 1))
            if return_weights:
                all_weights.append(weights)

        accumulator = self._weight_summary_accumulator
        if accumulator is None:
            self._set_weight_summary(weight_sum, weight_square_sum, weight_count)
        else:
            accumulator["sum"] += weight_sum
            accumulator["square_sum"] += weight_square_sum
            accumulator["count"] += weight_count

        if return_weights:
            return fused_score, torch.cat(all_weights, dim=0)
        return fused_score

    def get_weight_summary(self):
        return self._last_weight_summary
