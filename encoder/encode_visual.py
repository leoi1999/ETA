#!/usr/bin/env python3
"""Encode entity images with ETA stability-aware patch reweighting.

The output mirrors the attribute CLIP cache shape so it can be consumed by a
trainable visual head later:
  {
      "entity_embeddings": {entity_id: np.ndarray(float32)},
      "entity_image_paths": {entity_id: [str, ...]},
      "entity_names": {entity_id: str},
      "metadata": {...}
  }
"""

import argparse
import json
import os
import pickle
import queue
import random
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


SAFE_CHAR_PATTERN = re.compile(r"[^0-9A-Za-z._-]+")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
PROJECT_ROOT = os.path.abspath(os.path.join(REPO_ROOT, ".."))


def build_run_dir(repo_root, dataset_name, modalities, root_name="logs"):
    """Create a timestamped encoder output directory without importing src."""
    from datetime import datetime

    modality_name = "_".join(str(item) for item in modalities) or "none"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = "{}-{}-{}".format(dataset_name, modality_name, timestamp)
    output_dir = os.path.join(repo_root, root_name, run_name)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


DEFAULT_CLIP_DIR = os.path.join(
    REPO_ROOT, "pretrained_models", "clip-vit-base-patch16"
)
DEFAULT_ENCODER_NAME = "visual"

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)


def log(message):
    print(message, flush=True)


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Precompute CLIP image embeddings in a single CLIP forward pass. "
            "Stability-aware patch weights are computed before selected Transformer blocks "
            "and applied as source-token attention weights inside those blocks."
        )
    )
    parser.add_argument(
        "--dataset-dir",
        required=True,
        help="dataset directory containing ent_ids_* and images/",
    )
    parser.add_argument(
        "--images-dir",
        default=None,
        help="image directory; defaults to <dataset-dir>/images",
    )
    parser.add_argument(
        "--clip-dir",
        default=DEFAULT_CLIP_DIR,
        help="local HuggingFace CLIP model directory",
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument(
        "--device",
        default=None,
        help="encoding device; defaults to CUDA when available, otherwise CPU",
    )
    parser.add_argument(
        "--device-ids",
        default=None,
        help="comma-separated CUDA ids for multi-GPU inference, e.g. 0,1",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=8,
        help="threads for image loading; 0 means load images in the main thread",
    )
    parser.add_argument(
        "--prefetch-batches",
        type=int,
        default=4,
        help="number of CPU-prepared batches to keep ahead of GPU encoding",
    )
    parser.add_argument(
        "--stats-interval",
        type=int,
        default=20,
        help="print pipeline timing stats every N consumed batches; 0 disables",
    )
    parser.add_argument(
        "--max-images-per-entity",
        type=int,
        default=0,
        help="limit images per entity; 0 means use all available images",
    )
    parser.add_argument(
        "--images-per-entity",
        dest="sample_images_per_entity",
        type=int,
        default=1,
        help=(
            "randomly sample N images per entity before encoding; "
            "0 means use all available images"
        ),
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=None,
        help="optional seed for image sampling; omitted means non-deterministic sampling",
    )
    parser.add_argument(
        "--max-entities",
        type=int,
        default=0,
        help="limit encoded entities for smoke tests; 0 means use all entities",
    )
    parser.add_argument(
        "--no-normalize",
        action="store_true",
        help="do not L2-normalize image features before entity pooling",
    )
    parser.add_argument(
        "--pooling",
        choices=("mean", "first"),
        default="mean",
        help="how to pool multiple image embeddings for one entity",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="run CLIP in float16 when using CUDA",
    )
    parser.add_argument(
        "--stability-top-k",
        type=int,
        default=20,
        help="number of most stable patch tokens selected per channel when accumulating votes",
    )
    parser.add_argument(
        "--stability-layers",
        default="2",
        help="comma-separated 1-based transformer block indices that use stability-aware patch reweighting attention weighting",
    )
    parser.add_argument(
        "--gaussian-low-pass-sigma",
        type=float,
        default=None,
        help=(
            "Gaussian low-pass sigma along the channel-frequency axis. "
            "Defaults to sqrt(hidden_dim)."
        ),
    )
    parser.add_argument(
        "--stability-epsilon",
        type=float,
        default=1e-6,
        help="numerical epsilon used in the stability score and patch normalization",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "optional output directory; defaults to "
            "<dataset-dir>/image"
        ),
    )
    parser.add_argument(
        "--output-name",
        default=None,
        help="optional pickle filename; defaults to image.pkl",
    )
    return parser


def sanitize_entity_id(entity_id):
    value = entity_id.strip().strip("<>")
    value = value.replace("http://", "http_").replace("https://", "https_")
    value = SAFE_CHAR_PATTERN.sub("_", value)
    return value.strip("_") or "entity"


def read_entity_ids(path):
    id_to_name = {}
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            entity_id, entity_name = line.split("\t", 1)
            id_to_name[int(entity_id)] = entity_name
    return id_to_name


def read_entity_dir_map(images_dir):
    path = Path(images_dir) / "entity_id_map.tsv"
    entity_to_dir = {}
    if not path.exists():
        return entity_to_dir

    with path.open("r", encoding="utf-8") as handle:
        header = next(handle, "").strip().split("\t")
        has_header = header[:2] == ["entity_id", "sanitized_dir"]
        if not has_header and len(header) >= 2:
            entity_to_dir[header[0]] = header[1]
        for line in handle:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) >= 2:
                entity_to_dir[parts[0]] = parts[1]
    return entity_to_dir


def resolve_entity_image_dir(images_root, entity_id, entity_name, entity_to_dir):
    """Resolve both legacy and kg-prefixed per-entity image directories."""
    sanitized_name = sanitize_entity_id(entity_name)
    mapped_dir = entity_to_dir.get(str(entity_id)) or entity_to_dir.get(entity_name)
    candidate_names = []
    if mapped_dir:
        candidate_names.append(mapped_dir)
    candidate_names.extend(
        (
            "kg1_{}_{}".format(entity_id, sanitized_name),
            "kg2_{}_{}".format(entity_id, sanitized_name),
            sanitized_name,
        )
    )

    seen = set()
    for dirname in candidate_names:
        if dirname in seen:
            continue
        seen.add(dirname)
        entity_dir = images_root / dirname
        if entity_dir.is_dir():
            return entity_dir
    return None


def collect_entity_images(
    entity_names,
    images_dir,
    max_images_per_entity,
    sample_images_per_entity,
    max_entities,
    rng,
):
    images_root = Path(images_dir)
    entity_to_dir = read_entity_dir_map(images_root)
    entity_image_paths = {}

    for entity_id, entity_name in sorted(entity_names.items()):
        entity_dir = resolve_entity_image_dir(
            images_root=images_root,
            entity_id=entity_id,
            entity_name=entity_name,
            entity_to_dir=entity_to_dir,
        )
        if entity_dir is None:
            continue

        paths = [
            str(path)
            for path in sorted(entity_dir.iterdir())
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        ]
        if max_images_per_entity > 0:
            paths = paths[:max_images_per_entity]
        if sample_images_per_entity > 0 and len(paths) > sample_images_per_entity:
            paths = rng.sample(paths, sample_images_per_entity)
        if paths:
            entity_image_paths[entity_id] = paths
            if max_entities > 0 and len(entity_image_paths) >= max_entities:
                break

    return entity_image_paths


def batched(items, batch_size):
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def load_image_item(item):
    import numpy as np
    from PIL import Image, UnidentifiedImageError

    entity_id, path = item
    try:
        with Image.open(path) as image:
            image.load()
            rgb_image = image.convert("RGB")
            image_array = np.asarray(rgb_image, dtype=np.uint8)
            if image_array.ndim != 3 or image_array.shape[2] != 3:
                raise ValueError("expected RGB image array, got shape {}".format(image_array.shape))
            return {
                "entity_id": entity_id,
                "path": path,
                "image": image_array,
                "error": None,
            }
    except (OSError, UnidentifiedImageError, ValueError) as exc:
        return {
            "entity_id": entity_id,
            "path": path,
            "image": None,
            "error": str(exc),
        }


def load_image_batch(batch, executor):
    if executor is None:
        loaded = [load_image_item(item) for item in batch]
    else:
        loaded = list(executor.map(load_image_item, batch))

    images = []
    valid_items = []
    errors = []
    for item in loaded:
        if item["error"] is not None:
            errors.append({"path": item["path"], "error": item["error"]})
            continue
        images.append(item["image"])
        valid_items.append((item["entity_id"], item["path"]))

    return images, valid_items, errors


def preprocess_images(processor, images, valid_items):
    try:
        inputs = processor(images=images, return_tensors="pt")
        return inputs["pixel_values"], valid_items, []
    except (TypeError, ValueError) as exc:
        pixel_values = []
        kept_items = []
        errors = []
        for image, item in zip(images, valid_items):
            entity_id, path = item
            try:
                inputs = processor(images=image, return_tensors="pt")
                pixel_values.append(inputs["pixel_values"][0])
                kept_items.append(item)
            except (TypeError, ValueError) as item_exc:
                errors.append(
                    {
                        "path": path,
                        "error": "processor failed after batch error '{}': {}".format(
                            exc,
                            item_exc,
                        ),
                    }
                )
        if not pixel_values:
            return None, kept_items, errors

        import torch

        return torch.stack(pixel_values, dim=0), kept_items, errors


def prepare_image_batch(batch, executor, processor):
    images, valid_items, errors = load_image_batch(batch, executor)
    if not images:
        return {
            "pixel_values": None,
            "valid_items": valid_items,
            "errors": errors,
            "image_count": 0,
        }

    pixel_values, valid_items, processor_errors = preprocess_images(
        processor=processor,
        images=images,
        valid_items=valid_items,
    )
    errors.extend(processor_errors)
    return {
        "pixel_values": pixel_values,
        "valid_items": valid_items,
        "errors": errors,
        "image_count": 0 if pixel_values is None else int(pixel_values.size(0)),
    }


def parse_device_ids(device_ids):
    if device_ids is None or device_ids == "":
        return []
    return [int(part.strip()) for part in device_ids.split(",") if part.strip()]


def parse_stability_layers(stability_layers):
    layers = []
    for part in stability_layers.split(","):
        part = part.strip()
        if not part:
            continue
        layer_index = int(part)
        if layer_index <= 0:
            raise ValueError("--stability-layers must contain positive 1-based indices")
        layers.append(layer_index)
    if not layers:
        raise ValueError("--stability-layers must contain at least one layer index")
    return sorted(set(layers))


def sanitize_filename_part(value):
    value = SAFE_CHAR_PATTERN.sub("_", value.strip())
    return value.strip("_") or "default"


def build_default_output_name(encoder_name, images_dir, stability_layers, stability_top_k):
    del encoder_name, images_dir, stability_layers, stability_top_k
    return "image.pkl"


def build_default_output_dir(dataset_dir, encoder_name, images_dir, stability_layers, stability_top_k):
    del encoder_name, images_dir, stability_layers, stability_top_k
    return os.path.join(dataset_dir, "image")


def encode_images(
    entity_image_paths,
    clip_dir,
    device,
    device_ids,
    batch_size,
    num_workers,
    prefetch_batches,
    stats_interval,
    normalize,
    pooling,
    fp16,
    stability_layers,
    stability_top_k,
    gaussian_low_pass_sigma,
    stability_epsilon,
):
    import numpy as np
    import torch
    import torch.nn.functional as F
    from PIL import ImageFile
    from tqdm import tqdm
    from transformers import CLIPImageProcessor, CLIPModel

    ImageFile.LOAD_TRUNCATED_IMAGES = True

    log("loading CLIP image processor: {}".format(clip_dir))
    processor = CLIPImageProcessor.from_pretrained(clip_dir, local_files_only=True)
    log("loading CLIP model: {}".format(clip_dir))
    model = CLIPModel.from_pretrained(clip_dir, local_files_only=True)
    model.eval()
    num_vision_layers = len(model.vision_model.encoder.layers)
    invalid_layers = [layer_index for layer_index in stability_layers if layer_index > num_vision_layers]
    if invalid_layers:
        raise ValueError(
            "stability-aware layer indices {} exceed CLIP vision layer count {}".format(
                invalid_layers,
                num_vision_layers,
            )
        )

    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    runtime_device = torch.device(device)
    if runtime_device.type == "cuda" and not torch.cuda.is_available():
        log(
            (
                "requested device {} but torch.cuda.is_available() is false; "
                "falling back to cpu"
            ).format(device)
        )
        log("torch version: {}".format(torch.__version__))
        log("torch CUDA build: {}".format(torch.version.cuda))
        log("CUDA_VISIBLE_DEVICES: {}".format(os.environ.get("CUDA_VISIBLE_DEVICES")))
        runtime_device = torch.device("cpu")
        device_ids = []
    elif runtime_device.type == "cuda":
        available_count = torch.cuda.device_count()
        if not device_ids:
            device_ids = [runtime_device.index if runtime_device.index is not None else 0]
        device_ids = [device_id for device_id in device_ids if device_id < available_count]
        if not device_ids:
            raise RuntimeError(
                "No valid CUDA device ids found. Requested {}, available count {}.".format(
                    device_ids,
                    available_count,
                )
            )
        runtime_device = torch.device("cuda:{}".format(device_ids[0]))
        log("CUDA devices visible: {}".format(available_count))
        for device_id in device_ids:
            log("using cuda:{} - {}".format(device_id, torch.cuda.get_device_name(device_id)))
    model = model.to(runtime_device)
    if fp16 and runtime_device.type == "cuda":
        model = model.half()
    log("runtime device: {}".format(runtime_device))

    if __package__:
        from .stability_clip import StabilityAwareClipImageEncoder
    else:
        from stability_clip import StabilityAwareClipImageEncoder

    if stability_top_k <= 0:
        raise ValueError("--stability-top-k must be > 0")
    if gaussian_low_pass_sigma is not None and gaussian_low_pass_sigma <= 0:
        raise ValueError("--gaussian-low-pass-sigma must be > 0")
    encoder = StabilityAwareClipImageEncoder(
        clip_model=model,
        stability_layer_indices=stability_layers,
        top_k=stability_top_k,
        sigma=gaussian_low_pass_sigma,
        eps=stability_epsilon,
    )
    if runtime_device.type == "cuda" and len(device_ids) > 1:
        encoder = torch.nn.DataParallel(encoder, device_ids=device_ids)
        log("multi-GPU DataParallel enabled on device ids: {}".format(device_ids))

    work_items = []
    for entity_id, paths in sorted(entity_image_paths.items()):
        if pooling == "first":
            paths = paths[:1]
        for path in paths:
            work_items.append((entity_id, path))

    log("entities scheduled for encoding: {}".format(len(entity_image_paths)))
    log("images scheduled for encoding: {}".format(len(work_items)))
    log("batch size: {}".format(batch_size))
    log("image loading threads: {}".format(num_workers))
    log("prefetch batches: {}".format(prefetch_batches))
    log("stats interval: {}".format(stats_interval))
    log("stability-aware layers: {}".format(stability_layers))
    log("patches selected per channel: {}".format(stability_top_k))
    log("Gaussian low-pass sigma: {}".format(gaussian_low_pass_sigma if gaussian_low_pass_sigma is not None else "sqrt(hidden_dim)"))
    log("stability epsilon: {}".format(stability_epsilon))

    per_entity_features = {}
    used_paths = {}
    failed_paths = []
    total_batches = (len(work_items) + batch_size - 1) // batch_size
    prefetch_batches = max(1, prefetch_batches)

    with torch.inference_mode():
        executor = ThreadPoolExecutor(max_workers=num_workers) if num_workers > 0 else None
        prepared_queue = queue.Queue(maxsize=prefetch_batches)
        stop_event = threading.Event()

        def producer():
            try:
                for batch in batched(work_items, batch_size):
                    if stop_event.is_set():
                        break
                    start_time = time.perf_counter()
                    prepared = prepare_image_batch(
                        batch=batch,
                        executor=executor,
                        processor=processor,
                    )
                    prepared["prepare_seconds"] = time.perf_counter() - start_time
                    prepared_queue.put(prepared)
            except BaseException as exc:
                prepared_queue.put({"error": exc})
            finally:
                prepared_queue.put(None)

        producer_thread = threading.Thread(target=producer, name="image-prefetcher")
        producer_thread.daemon = True
        producer_thread.start()

        wait_seconds = 0.0
        gpu_seconds = 0.0
        prepare_seconds = 0.0
        consumed_batches = 0
        consumed_images = 0
        try:
            progress = tqdm(
                total=total_batches,
                desc="Encoding entity images",
                file=sys.stdout,
            )
            try:
                while True:
                    wait_start = time.perf_counter()
                    prepared = prepared_queue.get()
                    wait_seconds += time.perf_counter() - wait_start
                    if prepared is None:
                        break
                    if "error" in prepared:
                        raise prepared["error"]

                    progress.update(1)
                    consumed_batches += 1
                    prepare_seconds += prepared.get("prepare_seconds", 0.0)
                    failed_paths.extend(prepared["errors"])
                    pixel_values = prepared["pixel_values"]
                    if pixel_values is None:
                        continue

                    pixel_values = pixel_values.to(runtime_device, non_blocking=True)
                    consumed_images += prepared["image_count"]
                    gpu_start = time.perf_counter()
                    if fp16 and runtime_device.type == "cuda":
                        pixel_values = pixel_values.half()
                    features = encoder(pixel_values)
                    if normalize:
                        features = F.normalize(features, dim=-1)
                    features = features.detach().cpu().float()
                    gpu_seconds += time.perf_counter() - gpu_start

                    for (entity_id, path), feature in zip(prepared["valid_items"], features):
                        per_entity_features.setdefault(entity_id, []).append(feature)
                        used_paths.setdefault(entity_id, []).append(path)

                    if stats_interval > 0 and consumed_batches % stats_interval == 0:
                        log(
                            (
                                "pipeline stats batches={} images={} "
                                "avg_prepare={:.3f}s avg_gpu={:.3f}s "
                                "avg_gpu_wait={:.3f}s"
                            ).format(
                                consumed_batches,
                                consumed_images,
                                prepare_seconds / max(consumed_batches, 1),
                                gpu_seconds / max(consumed_batches, 1),
                                wait_seconds / max(consumed_batches, 1),
                            )
                        )
            finally:
                progress.close()
        finally:
            stop_event.set()
            producer_thread.join(timeout=5.0)
            if executor is not None:
                executor.shutdown(wait=True)

    entity_embeddings = {}
    for entity_id, features in per_entity_features.items():
        stacked = torch.stack(features, dim=0)
        if pooling == "mean":
            pooled = stacked.mean(dim=0)
            if normalize:
                pooled = F.normalize(pooled, dim=0)
        else:
            pooled = stacked[0]
        entity_embeddings[entity_id] = pooled.numpy().astype(np.float32)

    return entity_embeddings, used_paths, failed_paths, str(runtime_device)


def run_encode_job(args, dataset_dir, entity_names, images_dir, output_dir):
    clip_dir = os.path.abspath(args.clip_dir)
    os.makedirs(output_dir, exist_ok=True)

    log("=" * 80)
    log("dataset_dir: {}".format(dataset_dir))
    log("images_dir: {}".format(images_dir))
    log("clip_dir: {}".format(clip_dir))
    log("output_dir: {}".format(output_dir))
    log("pooling: {}".format(args.pooling))
    log("normalize: {}".format(not args.no_normalize))
    log("max_images_per_entity: {}".format(args.max_images_per_entity))
    log("sample_images_per_entity: {}".format(args.sample_images_per_entity))
    log("random_seed: {}".format(args.random_seed))
    log("max_entities: {}".format(args.max_entities))
    log("stability-aware layers: {}".format(args.stability_layers))
    log("patches selected per channel: {}".format(args.stability_top_k))
    if args.max_images_per_entity < 0:
        raise ValueError("--max-images-per-entity must be >= 0")
    if args.sample_images_per_entity < 0:
        raise ValueError("--images-per-entity must be >= 0")
    if args.stability_top_k <= 0:
        raise ValueError("--stability-top-k must be > 0")
    if args.gaussian_low_pass_sigma is not None and args.gaussian_low_pass_sigma <= 0:
        raise ValueError("--gaussian-low-pass-sigma must be > 0")
    if args.stability_epsilon <= 0:
        raise ValueError("--stability-epsilon must be > 0")
    device_ids = parse_device_ids(args.device_ids)
    log("requested device: {}".format(args.device))
    log("requested device ids: {}".format(device_ids if device_ids else "auto"))

    entity_image_paths = collect_entity_images(
        entity_names=entity_names,
        images_dir=images_dir,
        max_images_per_entity=args.max_images_per_entity,
        sample_images_per_entity=args.sample_images_per_entity,
        max_entities=args.max_entities,
        rng=random.Random(args.random_seed),
    )
    log("entities with local images: {}".format(len(entity_image_paths)))
    if entity_names:
        log(
            "image entity coverage: {:.2f}%".format(
                100.0 * len(entity_image_paths) / len(entity_names)
            )
        )
    if not entity_image_paths:
        raise RuntimeError(
            "No entity images were found under {!r}. Expected per-entity directories "
            "named kg1_<entity_id>_<sanitized_name>, "
            "kg2_<entity_id>_<sanitized_name>, or <sanitized_name>.".format(images_dir)
        )
    entity_embeddings, used_paths, failed_paths, runtime_device = encode_images(
        entity_image_paths=entity_image_paths,
        clip_dir=clip_dir,
        device=args.device,
        device_ids=device_ids,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        prefetch_batches=args.prefetch_batches,
        stats_interval=args.stats_interval,
        normalize=not args.no_normalize,
        pooling=args.pooling,
        fp16=args.fp16,
        stability_layers=args.stability_layers,
        stability_top_k=args.stability_top_k,
        gaussian_low_pass_sigma=args.gaussian_low_pass_sigma,
        stability_epsilon=args.stability_epsilon,
    )

    output_name = args.output_name or build_default_output_name(
        encoder_name="clip",
        images_dir=images_dir,
        stability_layers=args.stability_layers,
        stability_top_k=args.stability_top_k,
    )
    if os.path.basename(output_name) != output_name:
        raise ValueError("--output-name must be a filename, not a path")
    if not output_name.endswith(".pkl"):
        raise ValueError("--output-name must end with .pkl")
    output_stem = os.path.splitext(output_name)[0]

    output = {
        "entity_embeddings": entity_embeddings,
        "entity_image_paths": used_paths,
        "entity_names": entity_names,
        "metadata": {
            "dataset_dir": dataset_dir,
            "images_dir": images_dir,
            "clip_dir": clip_dir,
            "output_dir": output_dir,
            "run_name": os.path.basename(os.path.normpath(output_dir)),
            "output_name": output_name,
            "entity_count": len(entity_embeddings),
            "entity_total": len(entity_names),
            "image_count": sum(len(paths) for paths in used_paths.values()),
            "failed_image_count": len(failed_paths),
            "embedding_dim": int(next(iter(entity_embeddings.values())).shape[0])
            if entity_embeddings
            else 0,
            "device": runtime_device,
            "device_ids": device_ids,
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "prefetch_batches": args.prefetch_batches,
            "stats_interval": args.stats_interval,
            "normalized": not args.no_normalize,
            "pooling": args.pooling,
            "max_images_per_entity": args.max_images_per_entity,
            "sample_images_per_entity": args.sample_images_per_entity,
            "random_seed": args.random_seed,
            "max_entities": args.max_entities,
            "stability_layers": args.stability_layers,
            "stability_top_k": args.stability_top_k,
            "patch_weight_rule": "normalized_votes",
            "gaussian_low_pass_sigma": args.gaussian_low_pass_sigma,
            "stability_epsilon": args.stability_epsilon,
            "stability_output_mode": "single_pass_attention_weighting",
            "stability_layer_numbering": "1-based transformer block indices",
        },
    }
    output_path = os.path.join(output_dir, output_name)
    with open(output_path, "wb") as handle:
        pickle.dump(output, handle, protocol=pickle.HIGHEST_PROTOCOL)

    metadata_path = os.path.join(output_dir, "{}.metadata.json".format(output_stem))
    with open(metadata_path, "w", encoding="utf-8") as handle:
        json.dump(output["metadata"], handle, ensure_ascii=False, indent=2)

    if failed_paths:
        failed_path = os.path.join(output_dir, "{}.failed_paths.json".format(output_stem))
        with open(failed_path, "w", encoding="utf-8") as handle:
            json.dump(failed_paths, handle, ensure_ascii=False, indent=2)
        log("failed paths: {}".format(failed_path))

    log("saved: {}".format(output_path))
    log("metadata: {}".format(metadata_path))
    log("entities: {}".format(output["metadata"]["entity_count"]))
    log("images: {}".format(output["metadata"]["image_count"]))
    log("failed images: {}".format(output["metadata"]["failed_image_count"]))
    log("embedding_dim: {}".format(output["metadata"]["embedding_dim"]))


def main():
    args = build_parser().parse_args()
    args.stability_layers = parse_stability_layers(args.stability_layers)
    dataset_dir = os.path.abspath(args.dataset_dir)

    log("process id: {}".format(os.getpid()))
    log("python executable: {}".format(sys.executable))

    if args.max_images_per_entity < 0:
        raise ValueError("--max-images-per-entity must be >= 0")
    if args.sample_images_per_entity < 0:
        raise ValueError("--images-per-entity must be >= 0")

    entity_names = {}
    entity_names.update(read_entity_ids(os.path.join(dataset_dir, "ent_ids_1")))
    entity_names.update(read_entity_ids(os.path.join(dataset_dir, "ent_ids_2")))
    log("loaded entity names: {}".format(len(entity_names)))

    images_dir = os.path.abspath(args.images_dir or os.path.join(dataset_dir, "images"))
    output_dir = (
        os.path.abspath(args.output_dir)
        if args.output_dir
        else build_default_output_dir(
            dataset_dir=dataset_dir,
            encoder_name="clip",
            images_dir=images_dir,
            stability_layers=args.stability_layers,
            stability_top_k=args.stability_top_k,
        )
    )
    run_encode_job(
        args=args,
        dataset_dir=dataset_dir,
        entity_names=entity_names,
        images_dir=images_dir,
        output_dir=output_dir,
    )


if __name__ == "__main__":
    main()
