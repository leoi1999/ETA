#!/usr/bin/env python3
"""Encode entity attribute text with a Hugging Face encoder for ``src``.

The default remains the original ``name is value`` plus CLS configuration.  A
schema-only ablation is available through ``--text-content names-only``; it
sorts and deduplicates the canonical attribute names and never places entity
names or attribute values in the BERT input.  Both modes store normalized
entity-level embeddings in the pickle format consumed by
``src/model/attribute/module.py``.

An optional development-set score export is provided for standalone analysis.
The exported score is raw cosine, matching equations (20)--(21) and the score
consumed by the runtime attribute module.
"""

import argparse
import json
import os
import pickle
import re
import sys

import numpy as np
import torch


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
PROJECT_ROOT = os.path.abspath(os.path.join(REPO_ROOT, ".."))

sys.path.insert(0, REPO_ROOT)
from src.utils.paths import build_run_dir  # noqa: E402

DEFAULT_MODEL_DIR = os.path.join(REPO_ROOT, "pretrained_models", "bert")
DEFAULT_ENCODER_NAME = "attribute"


def build_default_output_dir(dataset_dir):
    """Return ``logs/<dataset>-attribute-<timestamp>``."""
    return build_run_dir(
        repo_root=REPO_ROOT,
        dataset_name=os.path.basename(os.path.normpath(dataset_dir)),
        modalities=(DEFAULT_ENCODER_NAME,),
    )


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Encode entity attributes as normalized Transformer vectors "
            "using either name-value text or a names-only schema sequence."
        )
    )
    parser.add_argument(
        "--dataset-dir",
        required=True,
        help="dataset directory containing ent_ids_1 and ent_ids_2",
    )
    parser.add_argument(
        "--attribute-file",
        default=None,
        help="attribute file; defaults to <dataset-dir>/attr",
    )
    parser.add_argument(
        "--model-dir",
        default=DEFAULT_MODEL_DIR,
        help="local Hugging Face encoder directory (for example BERT or RoBERTa)",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="output directory; defaults to <repo>/logs/<dataset>-attribute-<timestamp>",
    )
    parser.add_argument(
        "--output-name",
        default=None,
        help="pickle filename; defaults according to text content and pooling",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="encoding device; defaults to CUDA when available, otherwise CPU",
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument(
        "--max-length",
        type=int,
        default=128,
        help="maximum BERT input length",
    )
    parser.add_argument(
        "--separator",
        default=None,
        help=(
            "text placed between adjacent fields; defaults to '; ' for "
            "name-value and ' [SEP] ' for names-only"
        ),
    )
    parser.add_argument(
        "--text-content",
        choices=("name-value", "names-only"),
        default="name-value",
        help="whether BERT sees attribute names and values or only names",
    )
    parser.add_argument(
        "--pooling",
        choices=("cls", "mean"),
        default="cls",
        help="pool the CLS token or mean-pool all non-padding tokens",
    )
    parser.add_argument(
        "--keep-raw-attribute-names",
        action="store_true",
        help="do not split camelCase/underscores in attribute names",
    )
    parser.add_argument(
        "--format-mode",
        choices=("basic", "unified"),
        default="basic",
        help=(
            "basic only humanizes names; unified additionally canonicalizes "
            "cross-KG aliases, dates, numeric units, and field order"
        ),
    )
    parser.add_argument(
        "--no-final-period",
        action="store_true",
        help="do not append a period to non-empty entity texts",
    )
    parser.add_argument(
        "--no-normalize",
        action="store_true",
        help="do not L2-normalize BERT embeddings",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="use float16 BERT weights; requires CUDA",
    )
    parser.add_argument(
        "--max-entities",
        type=int,
        default=0,
        help="encode only the first N entities for a smoke test; 0 means all",
    )
    parser.add_argument(
        "--export-text-jsonl",
        action="store_true",
        help="also save the exact entity texts passed to BERT",
    )
    parser.add_argument(
        "--export-dev-scores",
        action="store_true",
        help="also export the full raw-cosine score matrix for dev_ent_ids",
    )
    parser.add_argument(
        "--ratio",
        default="0.2",
        help="split ratio used by --export-dev-scores",
    )
    parser.add_argument(
        "--split-dir",
        default=None,
        help="directory containing dev_ent_ids; defaults to <dataset-dir>/seed<ratio>",
    )
    parser.add_argument(
        "--score-output-name",
        default="attr_name_value_bert_cls_score.pkl",
    )
    parser.add_argument(
        "--score-chunk-size",
        type=int,
        default=512,
        help="row chunk size used to compute the optional dev score matrix",
    )
    return parser


def read_entities(path, side):
    entities = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            entity_id, entity_name = line.split(None, 1)
            entities.append(
                {
                    "entity_id": int(entity_id),
                    "entity_name": entity_name,
                    "side": side,
                }
            )
    return entities


def normalize_attribute_name(name):
    name = name.strip().strip("<>")
    if "#" in name:
        name = name.rsplit("#", 1)[1]
    elif "/" in name:
        name = name.rsplit("/", 1)[1]
    name = name.replace("_", " ").replace("-", " ")
    name = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", name)
    return re.sub(r"\s+", " ", name).strip().lower()


def clean_attribute_value(value):
    value = value.strip().strip('"')
    return re.sub(r"\s+", " ", value).strip()


ATTRIBUTE_ALIASES = {
    "location.geocode.latitude": "latitude",
    "location.geocode.longitude": "longitude",
    "location.location.area": "area total",
    "geography.river.length": "length",
    "geography.river.discharge": "discharge",
    "architecture.structure.opened": "opening date",
    "architecture.museum.established": "founding date",
    "food.food.energy": "energy",
    "lat": "latitude",
    "long": "longitude",
    "population number": "population total",
    "total population": "population total",
    "date of birth": "birth date",
    "birth year": "birth date",
    "date of death": "death date",
    "death year": "death date",
    "founded": "founding date",
    "date founded": "founding date",
    "established": "founding date",
    "date established": "founding date",
    "founding year": "founding date",
    "initial release date": "release date",
    "formation year": "formation date",
    "extinction year": "extinction date",
    "dissolution year": "dissolution date",
    "active years start year": "active years start date",
    "active years end year": "active years end date",
    "service start year": "service start date",
    "service end year": "service end date",
    "production start year": "production start date",
    "first publication year": "first publication date",
    "height": "height meters",
    "episode running time": "runtime",
    "air date of first episode": "release date",
    "air date of final episode": "completion date",
    "active start": "active years start date",
    "active end": "active years end date",
    # YAGO15K uses relation-style names for several attributes that are
    # represented as ordinary Freebase/DBpedia fields on the other side.
    "has latitude": "latitude",
    "has longitude": "longitude",
    "was born on date": "birth date",
    "died on date": "death date",
    "was destroyed on date": "dissolution date",
    # YAGO's generic creation date is most often aligned with founding or
    # establishment dates in DB15K.  Keeping one shared canonical name is
    # more useful for cross-KG matching than leaving the YAGO-only phrase.
    "was created on date": "founding date",
    "date dissolved": "dissolution date",
    "ceased operations": "dissolution date",
}

DATE_ATTRIBUTE_NAMES = {
    "birth date",
    "death date",
    "founding date",
    "formation date",
    "extinction date",
    "dissolution date",
    "opening date",
    "release date",
    "completion date",
    "active years start date",
    "active years end date",
    "service start date",
    "service end date",
    "production start date",
    "first publication date",
    "last publication date",
    "start date",
    "end date",
    "record date",
    "election date leader",
    "last election date",
    "publication date",
    "introduction date",
    "building start date",
    "closing date",
    "canonized date",
    "foal date",
}

ATTRIBUTE_UNITS = {
    "area total": "square meters",
    "area land": "square meters",
    "area water": "square meters",
    "area metro": "square meters",
    "area urban": "square meters",
    "area rural": "square meters",
    "area of catchment": "square meters",
    "surface area": "square meters",
    "floor area": "square meters",
    "length": "meters",
    "runway length": "meters",
    "width": "meters",
    "mainspan": "meters",
    "shore length": "meters",
    "elevation": "meters",
    "minimum elevation": "meters",
    "maximum elevation": "meters",
    "government elevation": "meters",
    "source elevation": "meters",
    "mouth elevation": "meters",
    "source confluence elevation": "meters",
    "height meters": "meters",
    "latitude": "degrees",
    "longitude": "degrees",
    "runtime": "seconds",
    "population total": "people",
    "population urban": "people",
    "population metro": "people",
    "population rural": "people",
    "number of students": "people",
    "number of undergraduate students": "people",
    "number of postgraduate students": "people",
    "number of graduate students": "people",
    "number of doctoral students": "people",
    "faculty size": "people",
    "staff": "people",
    "number of staff": "people",
    "number of employees": "people",
}

SOURCE_VALUE_SCALES = {
    "location.location.area": 1_000_000.0,
    "geography.river.length": 1_000.0,
    "tv.tv_program.episode_running_time": 60.0,
}


def canonical_attribute_name(raw_name):
    raw_name = raw_name.strip().strip("<>")
    if raw_name in ATTRIBUTE_ALIASES:
        return ATTRIBUTE_ALIASES[raw_name]

    # For namespace-qualified Freebase predicates, the final component is the
    # actual attribute name.  Keeping ``people.person`` in front of
    # ``date_of_birth`` prevents it from matching DBpedia's ``birthDate``.
    name_for_normalization = raw_name
    if "." in raw_name and "://" not in raw_name:
        name_for_normalization = raw_name.rsplit(".", 1)[1]
    normalized = normalize_attribute_name(name_for_normalization)
    return ATTRIBUTE_ALIASES.get(normalized, normalized)


def format_numeric_value(value):
    number = float(value)
    if not np.isfinite(number):
        return str(value)
    if number == 0:
        return "0"
    if abs(number - round(number)) < 1e-6:
        return "{:.0f}".format(number)
    return np.format_float_positional(
        number,
        unique=True,
        precision=10,
        trim="-",
    )


def normalize_date_value(value):
    value = clean_attribute_value(value)
    iso_match = re.match(r"^(-?\d{1,4})-(\d{1,2})(?:-(\d{1,2}))?$", value)
    if iso_match:
        year = int(iso_match.group(1))
        month = int(iso_match.group(2))
        if 1 <= month <= 12:
            return "{:04d}-{:02d}".format(year, month)
        return str(year)

    # The YAGO numerical file encodes dates as YYYY.MMDD, for example
    # 1936.1024 means 1936-10-24 and 1869.0000 means 1869.  This is not a
    # fractional year, so it must be handled before the Freebase fractional
    # year conversion below.  The final representation intentionally keeps
    # year-month precision, matching Freebase's fractional-year conversion.
    compact_match = re.match(r"^(-?\d{1,4})\.(\d{2})(\d{2})$", value)
    if compact_match:
        year = int(compact_match.group(1))
        month = int(compact_match.group(2))
        day = int(compact_match.group(3))
        if month == 0 and day == 0:
            return str(year)
        if 1 <= month <= 12:
            if day == 0 or 1 <= day <= 31:
                return "{:04d}-{:02d}".format(year, month)

    try:
        number = float(value)
    except ValueError:
        return value
    if not (1000 <= number <= 3000):
        return format_numeric_value(value)
    year = int(np.floor(number))
    fraction = number - year
    month = int(round(fraction * 12))
    if 1 <= month <= 11:
        return "{:04d}-{:02d}".format(year, month)
    return str(year)


def canonical_attribute_value(raw_name, canonical_name, value, side):
    value = clean_attribute_value(value)
    if not value:
        return value
    if canonical_name in DATE_ATTRIBUTE_NAMES:
        return normalize_date_value(value)
    try:
        number = float(value)
    except ValueError:
        return value.lower()
    if side == "source":
        number *= SOURCE_VALUE_SCALES.get(raw_name, 1.0)
    formatted = format_numeric_value(number)
    unit = ATTRIBUTE_UNITS.get(canonical_name)
    return "{} {}".format(formatted, unit) if unit else formatted


def sort_and_deduplicate_records(records):
    unique = {}
    for record in records:
        unique[(record["name"], record["value"])] = record

    # Prefer a year-month value over a duplicate year-only value for the same
    # date attribute, since the source numerical triples often only retain the
    # month while the target side may contain a full ISO date.
    detailed_date_years = {
        (record["name"], record["value"].split("-", 1)[0])
        for record in unique.values()
        if record["name"] in DATE_ATTRIBUTE_NAMES
        and re.match(r"^-?\d{1,4}-\d{2}$", record["value"])
    }
    filtered = []
    for record in unique.values():
        if (
            record["name"] in DATE_ATTRIBUTE_NAMES
            and re.match(r"^-?\d{1,4}$", record["value"])
            and (record["name"], record["value"]) in detailed_date_years
        ):
            continue
        filtered.append(record)
    return sorted(filtered, key=lambda item: (item["name"], item["value"]))


def parse_attribute_line(
    line,
    keep_raw_attribute_names=False,
    format_mode="basic",
    side=None,
):
    records = []
    if not line.strip():
        return records

    for chunk in line.rstrip("\n").split("^^^"):
        if not chunk:
            continue
        parts = chunk.split("|||")
        if len(parts) < 2:
            continue
        raw_name = parts[0].strip()
        if format_mode == "unified":
            name = canonical_attribute_name(raw_name)
            value = canonical_attribute_value(raw_name, name, parts[1], side)
        else:
            name = (
                raw_name
                if keep_raw_attribute_names
                else normalize_attribute_name(raw_name)
            )
            value = clean_attribute_value(parts[1])
        datatype = parts[2].strip() if len(parts) >= 3 else ""
        if not name:
            continue
        records.append(
            {
                "name": name,
                "value": value,
                "datatype": datatype,
            }
        )
    if format_mode == "unified":
        return sort_and_deduplicate_records(records)
    return records


def build_name_value_text(records, separator="; ", append_period=True):
    fields = []
    for record in records:
        if record["value"]:
            fields.append("{} is {}".format(record["name"], record["value"]))
        else:
            fields.append(record["name"])
    text = separator.join(fields)
    if text and append_period:
        text += "."
    return text


def build_name_only_text(records, separator=" [SEP] ", append_period=False):
    """Build a deterministic schema sequence without values or entity names."""
    names = sorted({record["name"] for record in records if record["name"]})
    text = separator.join(names)
    if text and append_period:
        text += "."
    return text


def build_entity_text_records(
    dataset_dir,
    attribute_file,
    separator,
    keep_raw_attribute_names=False,
    format_mode="basic",
    append_period=True,
    max_entities=0,
    text_content="name-value",
):
    entities = read_entities(os.path.join(dataset_dir, "ent_ids_1"), "source")
    entities.extend(read_entities(os.path.join(dataset_dir, "ent_ids_2"), "target"))

    with open(attribute_file, "r", encoding="utf-8") as handle:
        attr_lines = handle.readlines()
    if len(attr_lines) != len(entities):
        raise ValueError(
            "attr line count {} does not match entity count {}".format(
                len(attr_lines),
                len(entities),
            )
        )

    if max_entities > 0:
        entities = entities[:max_entities]
        attr_lines = attr_lines[:max_entities]

    output = []
    for entity, attr_line in zip(entities, attr_lines):
        records = parse_attribute_line(
            attr_line,
            keep_raw_attribute_names=keep_raw_attribute_names,
            format_mode=format_mode,
            side=entity["side"],
        )
        if text_content == "names-only":
            attribute_text = build_name_only_text(
                records,
                separator=separator,
                append_period=append_period,
            )
            attribute_count = len({record["name"] for record in records})
        elif text_content == "name-value":
            attribute_text = build_name_value_text(
                records,
                separator=separator,
                append_period=append_period,
            )
            attribute_count = len(records)
        else:
            raise ValueError("unsupported text content: {}".format(text_content))
        item = dict(entity)
        item.update(
            {
                "attribute_text": attribute_text,
                "attribute_count": attribute_count,
                "has_attributes": bool(attribute_text),
            }
        )
        output.append(item)
    return output


def masked_mean(last_hidden_state, attention_mask):
    mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
    summed = torch.sum(last_hidden_state * mask, dim=1)
    counts = mask.sum(dim=1).clamp_min(1e-9)
    return summed / counts


@torch.inference_mode()
def encode_bert_embeddings(
    texts,
    model_dir,
    device,
    batch_size,
    max_length,
    pooling,
    normalize,
    fp16,
):
    from transformers import AutoModel, AutoTokenizer

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    runtime_device = torch.device(device)
    if runtime_device.type == "cuda" and not torch.cuda.is_available():
        runtime_device = torch.device("cpu")
    if fp16 and runtime_device.type != "cuda":
        raise ValueError("--fp16 requires a CUDA device")

    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    model = AutoModel.from_pretrained(model_dir, local_files_only=True)
    model.eval().to(runtime_device)
    if fp16:
        model.half()

    batches = []
    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start : start + batch_size]
        tokens = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        tokens = {
            key: value.to(runtime_device, non_blocking=True)
            for key, value in tokens.items()
        }
        last_hidden_state = model(**tokens).last_hidden_state
        if pooling == "cls":
            embeddings = last_hidden_state[:, 0].float()
        elif pooling == "mean":
            content_mask = tokens["attention_mask"].bool()
            for special_token_id in tokenizer.all_special_ids:
                content_mask &= tokens["input_ids"].ne(special_token_id)
            # This fallback is only relevant for a pathological all-special
            # input; normal non-empty attribute texts always retain content.
            empty_content = ~content_mask.any(dim=1)
            if torch.any(empty_content):
                content_mask[empty_content] = tokens["attention_mask"][empty_content].bool()
            embeddings = masked_mean(
                last_hidden_state,
                content_mask,
            ).float()
        else:
            raise ValueError("unsupported pooling: {}".format(pooling))
        if normalize:
            embeddings = torch.nn.functional.normalize(
                embeddings,
                p=2,
                dim=-1,
            )
        batches.append(embeddings.cpu().numpy().astype(np.float32))

    return np.concatenate(batches, axis=0), str(runtime_device)


def read_alignment_pairs(path):
    pairs = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            left_id, right_id = line.split()[:2]
            pairs.append((int(left_id), int(right_id)))
    return np.asarray(pairs, dtype=np.int64)


def compute_cosine_score_matrix(
    entity_embeddings,
    entity_embedding_validity,
    alignment_pairs,
    chunk_size=512,
):
    if chunk_size <= 0:
        raise ValueError("score chunk size must be positive")

    left_ids = alignment_pairs[:, 0]
    right_ids = alignment_pairs[:, 1]
    left_embeddings = np.stack(
        [entity_embeddings[int(entity_id)] for entity_id in left_ids],
        axis=0,
    ).astype(np.float32)
    right_embeddings = np.stack(
        [entity_embeddings[int(entity_id)] for entity_id in right_ids],
        axis=0,
    ).astype(np.float32)
    left_valid = np.asarray(
        [entity_embedding_validity[int(entity_id)] for entity_id in left_ids],
        dtype=np.bool_,
    )
    right_valid = np.asarray(
        [entity_embedding_validity[int(entity_id)] for entity_id in right_ids],
        dtype=np.bool_,
    )

    left_norm = np.linalg.norm(left_embeddings, axis=1, keepdims=True)
    right_norm = np.linalg.norm(right_embeddings, axis=1, keepdims=True)
    left_embeddings = left_embeddings / np.maximum(left_norm, 1e-12)
    right_embeddings = right_embeddings / np.maximum(right_norm, 1e-12)

    scores = np.zeros((len(left_ids), len(right_ids)), dtype=np.float32)
    for start in range(0, len(left_ids), chunk_size):
        end = min(start + chunk_size, len(left_ids))
        cosine = left_embeddings[start:end] @ right_embeddings.T
        cosine = np.clip(cosine, -1.0, 1.0)
        valid_chunk = left_valid[start:end, None] & right_valid[None, :]
        scores[start:end] = cosine * valid_chunk.astype(np.float32)
    return scores


def save_text_records(path, records):
    with open(path, "w", encoding="utf-8") as handle:
        for record in records:
            json.dump(record, handle, ensure_ascii=False)
            handle.write("\n")


def read_encoder_metadata(model_dir):
    """Read lightweight encoder identity fields without loading the model."""
    config_path = os.path.join(model_dir, "config.json")
    config = {}
    if os.path.isfile(config_path):
        with open(config_path, "r", encoding="utf-8") as handle:
            config = json.load(handle)
    return {
        "encoder_name": os.path.basename(os.path.normpath(model_dir)),
        "encoder_model_type": config.get("model_type", "unknown"),
        "encoder_architectures": config.get("architectures", []),
    }


def main():
    args = build_parser().parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.max_length <= 0:
        raise ValueError("--max-length must be positive")
    if args.max_entities < 0:
        raise ValueError("--max-entities must be non-negative")

    dataset_dir = os.path.abspath(args.dataset_dir)
    attribute_file = os.path.abspath(
        args.attribute_file or os.path.join(dataset_dir, "attr")
    )
    model_dir = os.path.abspath(args.model_dir)
    encoder_metadata = read_encoder_metadata(model_dir)
    output_dir = os.path.abspath(args.output_dir) if args.output_dir else build_default_output_dir(
        dataset_dir
    )
    os.makedirs(output_dir, exist_ok=True)
    separator = args.separator
    if separator is None:
        separator = " [SEP] " if args.text_content == "names-only" else "; "
    append_period = not args.no_final_period
    if args.text_content == "names-only" and not args.no_final_period:
        append_period = False
    output_name = args.output_name or "attr_{}_bert_{}_embeddings.pkl".format(
        args.text_content.replace("-", "_"),
        args.pooling,
    )

    records = build_entity_text_records(
        dataset_dir=dataset_dir,
        attribute_file=attribute_file,
        separator=separator,
        keep_raw_attribute_names=args.keep_raw_attribute_names,
        format_mode=args.format_mode,
        append_period=append_period,
        max_entities=args.max_entities,
        text_content=args.text_content,
    )
    valid_records = [record for record in records if record["has_attributes"]]
    if not valid_records:
        raise RuntimeError("No non-empty entity attribute texts were found")

    encoded, runtime_device = encode_bert_embeddings(
        texts=[record["attribute_text"] for record in valid_records],
        model_dir=model_dir,
        device=args.device,
        batch_size=args.batch_size,
        max_length=args.max_length,
        pooling=args.pooling,
        normalize=not args.no_normalize,
        fp16=args.fp16,
    )
    embedding_dim = int(encoded.shape[1])
    zero_embedding = np.zeros((embedding_dim,), dtype=np.float32)
    valid_embedding_by_id = {
        record["entity_id"]: encoded[index]
        for index, record in enumerate(valid_records)
    }
    entity_embeddings = {
        record["entity_id"]: valid_embedding_by_id.get(
            record["entity_id"],
            zero_embedding.copy(),
        )
        for record in records
    }
    entity_embedding_validity = {
        record["entity_id"]: bool(record["has_attributes"])
        for record in records
    }

    metadata = {
        "dataset_dir": dataset_dir,
        "attribute_file": attribute_file,
        "model_dir": model_dir,
        "output_dir": output_dir,
        "run_name": os.path.basename(os.path.normpath(output_dir)),
        **encoder_metadata,
        "device": runtime_device,
        "entity_count": len(records),
        "nonempty_entity_count": len(valid_records),
        "empty_entity_count": len(records) - len(valid_records),
        "attribute_record_count": int(
            sum(record["attribute_count"] for record in records)
        ),
        "embedding_dim": embedding_dim,
        "pooling": args.pooling,
        "max_length": args.max_length,
        "normalized": not args.no_normalize,
        "dtype": "float32",
        "representation": "entity_attribute_{}_{}_{}".format(
            args.text_content.replace("-", "_"),
            encoder_metadata["encoder_model_type"],
            args.pooling,
        ),
        "text_template": (
            "attribute name [SEP] attribute name ..."
            if args.text_content == "names-only"
            else "attribute name is attribute value; ..."
        ),
        "text_content": (
            "attribute_names_only"
            if args.text_content == "names-only"
            else "attribute_names_and_values_only"
        ),
        "entity_name_in_text": False,
        "separator": separator,
        "attribute_name_normalization": (
            "cross_kg_canonical_aliases"
            if args.format_mode == "unified"
            else ("raw" if args.keep_raw_attribute_names else "humanized_lowercase")
        ),
        "format_mode": args.format_mode,
        "record_order": (
            "deduplicated_attribute_name"
            if args.text_content == "names-only"
            else (
                "canonical_name_then_value"
                if args.format_mode == "unified"
                else "input"
            )
        ),
        "value_normalization": (
            "not_applicable_values_excluded"
            if args.text_content == "names-only"
            else (
                "dates_to_year_month_and_selected_units_to_si"
                if args.format_mode == "unified"
                else "whitespace_only"
            )
        ),
        "empty_attribute_policy": "zero_vector_with_invalid_mask",
        "runtime_score": "raw cosine",
        "projector_required": False,
        "projector_recommended": args.text_content == "names-only",
        "max_entities": args.max_entities,
    }
    output = {
        "entity_embeddings": entity_embeddings,
        "entity_embedding_validity": entity_embedding_validity,
        "metadata": metadata,
    }
    output_path = os.path.join(output_dir, output_name)
    with open(output_path, "wb") as handle:
        pickle.dump(output, handle, protocol=pickle.HIGHEST_PROTOCOL)

    metadata_path = os.path.splitext(output_path)[0] + ".metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    if args.export_text_jsonl:
        text_path = os.path.join(
            output_dir,
            "entity_attribute_{}_texts.jsonl".format(
                args.text_content.replace("-", "_")
            ),
        )
        save_text_records(text_path, records)
        print("texts:", text_path)

    if args.export_dev_scores:
        if args.max_entities > 0:
            raise ValueError(
                "--export-dev-scores cannot be combined with --max-entities"
            )
        split_dir = (
            os.path.abspath(args.split_dir)
            if args.split_dir
            else os.path.join(dataset_dir, "seed{}".format(args.ratio))
        )
        dev_pair_path = os.path.join(split_dir, "dev_ent_ids")
        dev_pairs = read_alignment_pairs(dev_pair_path)
        score_matrix = compute_cosine_score_matrix(
            entity_embeddings=entity_embeddings,
            entity_embedding_validity=entity_embedding_validity,
            alignment_pairs=dev_pairs,
            chunk_size=args.score_chunk_size,
        )
        score_metadata = dict(metadata)
        score_metadata.update(
            {
                "dev_ent_ids": dev_pair_path,
                "dev_pair_count": int(len(dev_pairs)),
                "score_shape": list(score_matrix.shape),
                "score_min": float(score_matrix.min()),
                "score_max": float(score_matrix.max()),
                "score_semantics": (
                    "rows use dev_ent_ids left IDs; columns use dev_ent_ids right IDs"
                ),
            }
        )
        score_output_path = os.path.join(output_dir, args.score_output_name)
        with open(score_output_path, "wb") as handle:
            pickle.dump(
                {
                    "precomputed_score_matrix": score_matrix,
                    "metadata": score_metadata,
                },
                handle,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
        score_metadata_path = os.path.splitext(score_output_path)[0] + ".metadata.json"
        with open(score_metadata_path, "w", encoding="utf-8") as handle:
            json.dump(score_metadata, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        print("dev scores:", score_output_path)

    print("saved:", output_path)
    print("metadata:", metadata_path)
    print("entities:", metadata["entity_count"])
    print("nonempty entities:", metadata["nonempty_entity_count"])
    print("empty entities:", metadata["empty_entity_count"])
    print("embedding_dim:", metadata["embedding_dim"])


if __name__ == "__main__":
    main()
