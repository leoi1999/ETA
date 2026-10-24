import json
import os
import time

import torch

try:
    from ..utils.evaluation import (
        NORMALIZATION_ROUNDS,
        NORMALIZATION_TEMPERATURE,
        evaluate_normalized_similarity,
    )
except ImportError:
    from utils.evaluation import (
        NORMALIZATION_ROUNDS,
        NORMALIZATION_TEMPERATURE,
        evaluate_normalized_similarity,
    )


class Evaluator:
    def __init__(
        self,
        model,
        logger,
        export_dev_predictions_path=None,
        export_graph_negatives=False,
        graph_negatives_dir=None,
        normalization_temperature=NORMALIZATION_TEMPERATURE,
        normalization_rounds=NORMALIZATION_ROUNDS,
    ):
        self.model = model
        self.logger = logger
        self.export_dev_predictions_path = export_dev_predictions_path
        self.export_graph_negatives = export_graph_negatives
        self.graph_negatives_dir = graph_negatives_dir
        self.normalization_temperature = float(normalization_temperature)
        self.normalization_rounds = int(normalization_rounds)
        if self.normalization_temperature <= 0:
            raise ValueError("normalization_temperature must be positive.")
        if self.normalization_rounds <= 0:
            raise ValueError("normalization_rounds must be positive.")
        if self.export_graph_negatives and not self.graph_negatives_dir:
            raise ValueError(
                "graph_negatives_dir is required when export_graph_negatives is enabled."
            )

    def evaluation_device(self):
        parameter = next(self.model.parameters(), None)
        if parameter is not None:
            return parameter.device
        buffer = next(self.model.buffers(), None)
        if buffer is not None:
            return buffer.device
        return torch.device("cpu")

    def log_evaluation_metrics(self, result, prefix="Evaluation metrics"):
        """Write the key ranking metrics to the training log."""
        if self.logger is None:
            return

        self.logger.info(
            "%s: Hits@1=%.4f Hits@3=%.4f Hits@10=%.4f MR=%.4f MRR=%.4f",
            prefix,
            float(result.get("hits@1", 0.0)),
            float(result.get("hits@3", 0.0)),
            float(result.get("hits@10", 0.0)),
            float(result.get("MR", 0.0)),
            float(result.get("MRR", 0.0)),
        )

    @torch.inference_mode()
    def encode_graph_features(self, dataset_state):
        if not self.model.has_graph():
            return None
        graph_embeddings = self.model.encode_graph_features(
            dataset_state=dataset_state,
        ).detach().cpu()
        torch.cuda.empty_cache()
        return graph_embeddings

    @torch.inference_mode()
    def evaluate_dev(self, dataset_state, graph_embeddings=None, return_details=False):
        left_entity_ids, right_entity_ids = dataset_state.dev_pairs.T
        evaluation_device = self.evaluation_device()
        graph_scores = self.model.compute_graph_similarity(
            dataset_state=dataset_state,
            source_entity_ids=left_entity_ids,
            target_entity_ids=right_entity_ids,
            graph_embeddings=graph_embeddings,
            output_device=evaluation_device,
        )
        if graph_scores is None and graph_embeddings is not None:
            left_indices = torch.as_tensor(left_entity_ids, dtype=torch.long)
            right_indices = torch.as_tensor(right_entity_ids, dtype=torch.long)
            left_vec = graph_embeddings[left_indices]
            right_vec = graph_embeddings[right_indices]
            left_vec = left_vec / (torch.linalg.norm(left_vec, dim=-1, keepdim=True) + 1e-5)
            right_vec = right_vec / (torch.linalg.norm(right_vec, dim=-1, keepdim=True) + 1e-5)
            graph_scores = (left_vec @ right_vec.T).to(evaluation_device)

        return evaluate_normalized_similarity(
            similarity_matrix=self.model.compute_fused_similarity(
                graph_scores=graph_scores
            ),
            len_dev=len(dataset_state.dev_pairs),
            return_details=return_details,
            normalization_temperature=self.normalization_temperature,
            normalization_rounds=self.normalization_rounds,
        )

    def export_dev_predictions(self, dataset_state, result, output_path):
        details = result.get("prediction_details")
        if not details:
            raise RuntimeError("Prediction details are unavailable; enable return_details for evaluation.")

        output_dir = os.path.dirname(os.path.abspath(output_path))
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        left_entity_ids = [int(value) for value in dataset_state.dev_pairs[:, 0].tolist()]
        right_entity_ids = [int(value) for value in dataset_state.dev_pairs[:, 1].tolist()]
        row_side = details["row_side"]
        col_side = details["col_side"]
        top1_indices = [int(value) for value in details["top1_indices"]]
        top1_scores = [float(value) for value in details["top1_scores"]]
        true_scores = [float(value) for value in details["true_scores"]]
        true_ranks = [int(value) for value in details["true_ranks"]]

        records = []
        for pair_index, predicted_index in enumerate(top1_indices):
            if row_side == "right":
                query_entity_id = right_entity_ids[pair_index]
                true_entity_id = left_entity_ids[pair_index]
                predicted_entity_id = left_entity_ids[predicted_index]
            else:
                query_entity_id = left_entity_ids[pair_index]
                true_entity_id = right_entity_ids[pair_index]
                predicted_entity_id = right_entity_ids[predicted_index]

            records.append(
                {
                    "pair_index": pair_index,
                    "left_entity_id": left_entity_ids[pair_index],
                    "right_entity_id": right_entity_ids[pair_index],
                    "row_side": row_side,
                    "col_side": col_side,
                    "query_entity_id": query_entity_id,
                    "true_entity_id": true_entity_id,
                    "predicted_pair_index": predicted_index,
                    "predicted_entity_id": predicted_entity_id,
                    "is_correct": predicted_index == pair_index,
                    "top1_score": top1_scores[pair_index],
                    "true_score": true_scores[pair_index],
                    "true_rank": true_ranks[pair_index],
                }
            )

        metrics = {
            key: float(value)
            for key, value in result.items()
            if key != "prediction_details"
        }
        payload = {
            "row_side": row_side,
            "col_side": col_side,
            "metrics": metrics,
            "records": records,
        }
        with open(output_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")

        if self.logger is not None:
            self.logger.info("Wrote per-dev-pair predictions to %s", output_path)

    def _graph_negative_output_path(self, epoch):
        if epoch is None:
            filename = "eval_only_graph_negatives.jsonl"
        else:
            filename = "epoch_{:04d}_graph_negatives.jsonl".format(epoch + 1)
        return os.path.join(self.graph_negatives_dir, filename)

    @torch.inference_mode()
    def export_graph_negatives_if_enabled(
        self,
        dataset_state,
        graph_embeddings,
        epoch=None,
    ):
        if not self.export_graph_negatives:
            return None
        if graph_embeddings is None:
            if self.logger is not None:
                self.logger.warning(
                    "Graph negative export skipped because graph embeddings are unavailable."
                )
            return None

        output_path = self._graph_negative_output_path(epoch)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        left_entity_ids, right_entity_ids = dataset_state.dev_pairs.T
        graph_scores = self.model.compute_graph_similarity(
            dataset_state=dataset_state,
            source_entity_ids=left_entity_ids,
            target_entity_ids=right_entity_ids,
            graph_embeddings=graph_embeddings,
        )
        if graph_scores is None:
            left_indices = torch.as_tensor(left_entity_ids, dtype=torch.long)
            right_indices = torch.as_tensor(right_entity_ids, dtype=torch.long)
            left_vec = graph_embeddings[left_indices]
            right_vec = graph_embeddings[right_indices]
            left_vec = left_vec / (torch.linalg.norm(left_vec, dim=-1, keepdim=True) + 1e-5)
            right_vec = right_vec / (torch.linalg.norm(right_vec, dim=-1, keepdim=True) + 1e-5)
            graph_scores = left_vec @ right_vec.T

        top1_scores, top1_indices = torch.max(graph_scores, dim=1)
        row_indices = torch.arange(len(dataset_state.dev_pairs), device=graph_scores.device)
        true_scores = graph_scores[row_indices, row_indices]
        true_ranks = torch.count_nonzero(
            graph_scores > true_scores.unsqueeze(1),
            dim=1,
        ) + 1

        negative_count = 0
        with open(output_path, "w", encoding="utf-8") as handle:
            for pair_index, predicted_index in enumerate(top1_indices.detach().cpu().tolist()):
                if predicted_index == pair_index:
                    continue
                record = {
                    "pair_index": int(pair_index),
                    "direction": "left_to_right",
                    "left_entity_id": int(left_entity_ids[pair_index]),
                    "true_right_entity_id": int(right_entity_ids[pair_index]),
                    "predicted_pair_index": int(predicted_index),
                    "predicted_right_entity_id": int(right_entity_ids[predicted_index]),
                    "top1_score": float(top1_scores[pair_index]),
                    "true_score": float(true_scores[pair_index]),
                    "true_rank": int(true_ranks[pair_index]),
                }
                json.dump(record, handle, ensure_ascii=False)
                handle.write("\n")
                negative_count += 1

        if self.logger is not None:
            self.logger.info(
                "Wrote %s graph negative examples to %s",
                negative_count,
                output_path,
            )
        return output_path

    def evaluate_epoch(
        self,
        dataset_state,
        epoch,
        total_epochs,
        require_graph_embeddings=False,
    ):
        started_at = time.time()
        self.logger.info("Running evaluation on %s validation pairs.", len(dataset_state.dev_pairs))
        self.model.eval()

        graph_embeddings = None
        if require_graph_embeddings:
            graph_embeddings = self.encode_graph_features(
                dataset_state=dataset_state
            )
        normalization_started_at = time.time()
        result = self.evaluate_dev(
            dataset_state=dataset_state,
            graph_embeddings=graph_embeddings,
            return_details=bool(self.export_dev_predictions_path),
        )
        normalization_ended_at = time.time()
        self.export_graph_negatives_if_enabled(
            dataset_state=dataset_state,
            graph_embeddings=graph_embeddings,
            epoch=epoch,
        )
        self.log_evaluation_metrics(result)
        if self.export_dev_predictions_path:
            self.export_dev_predictions(
                dataset_state=dataset_state,
                result=result,
                output_path=self.export_dev_predictions_path,
            )
        self.logger.info(
            "%s-round normalization evaluation time: %.4f seconds",
            self.normalization_rounds,
            normalization_ended_at - normalization_started_at,
        )

        ended_at = time.time()
        self.logger.info("Epoch %s/%s evaluation time: %.4f seconds", epoch + 1, total_epochs, ended_at - started_at)
        return graph_embeddings, result
