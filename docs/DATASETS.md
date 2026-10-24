# Datasets

## Download the exact prepared data

[Download eta-data-v1.zip](https://github.com/leoi1999/ETA/releases/download/data-v1/eta-data-v1.zip), or run `python tools/download_data.py` from the repository root. Extract the archive in the repository root, preserving the `data/` directory.

The archive contains both datasets and the visual features from the supplied ETA code archive. Raw images, attribute embeddings, model weights, and training checkpoints are not included.

- Archive size: 112,147,568 bytes (107.0 MiB).
- Files: 46.
- SHA-256: `f5c699ee6814ebdd8e3a63b1ebb68a37e99b30ea707961c57b3d0dbd054cc0c5`.

Individual file sizes and SHA-256 values are in [data_manifest.json](data_manifest.json). Check an existing installation using `python tools/download_data.py --check`.

## Dataset statistics

The following counts are measured from the released files rather than inferred from the benchmark names.

| Benchmark | Graph | Entities | Relations | Triples | Alignment pairs |
| --- | --- | ---: | ---: | ---: | ---: |
| FB15K-DB15K | Freebase | 14,951 | 1,345 | 592,213 | 12,846 |
| FB15K-DB15K | DBpedia | 12,842 | 279 | 99,028 | 12,846 |
| FB15K-YAGO15K | Freebase | 14,951 | 1,345 | 592,213 | 11,199 |
| FB15K-YAGO15K | YAGO | 15,404 | 32 | 122,886 | 11,199 |

### Provided non-iterative splits

| Benchmark | Seed ratio | Supervised pairs | Development pairs | Reference pairs |
| --- | ---: | ---: | ---: | ---: |
| FB15K-DB15K | 0.2 | 2,569 | 10,277 | 12,846 |
| FB15K-DB15K | 0.5 | 6,423 | 6,423 | 12,846 |
| FB15K-DB15K | 0.8 | 10,277 | 2,569 | 12,846 |
| FB15K-YAGO15K | 0.2 | 2,239 | 8,960 | 11,199 |
| FB15K-YAGO15K | 0.5 | 5,599 | 5,600 | 11,199 |
| FB15K-YAGO15K | 0.8 | 8,960 | 2,239 | 11,199 |

Pseudo-label experiments construct a separate train/validation/test split from the full reference pairs; see [evaluation protocols](REPRODUCIBILITY.md#evaluation-protocols).

## File layout and formats

```text
data/
|-- MMKB-LICENSE.txt
|-- FB15K_DB15K/
|   |-- ent_ids_1, ent_ids_2             # Integer ID + entity identifier
|   |-- triples_1, triples_2             # Tab-separated integer head/relation/tail
|   |-- attr                            # Attributes used by the BERT encoder
|   |-- training_attrs_1, training_attrs_2
|   |-- ill_ent_ids                     # Full entity alignment pairs
|   |-- seed0.2/, seed0.5/, seed0.8/
|   |   `-- sup_ent_ids, dev_ent_ids, ref_ent_ids
|   |-- image_manifest_template.tsv
|   `-- image/
|       |-- image.pkl                   # Supplied 512-dimensional visual features
|       `-- image.metadata.json
`-- FB15K_YAGO15K/                      # Equivalent layout and source YAGO files
```

Generated attribute features belong at `<dataset>/bert_attr_from_mmkb_unified/attr_name_value_bert_cls_embeddings.pkl`. Paths in the launchers are resolved relative to the dataset directory.

## Visual feature metadata

Both feature files use CLIP ViT-B/16, mean pooling, normalized 512-dimensional embeddings, stability layers 1/5/8, top-k 30, normalized-vote patch weighting, and image sampling seed 42, according to the included extraction metadata. The metadata is preserved unchanged.

| Benchmark | Entities with visual features | Total entities in extraction metadata |
| --- | ---: | ---: |
| FB15K-DB15K | 27,580 | 27,793 |
| FB15K-YAGO15K | 25,940 | 30,355 |

## Original data and related benchmarks

| Resource | Link | Use |
| --- | --- | --- |
| MMKG/MMKB | [Repository](https://github.com/mniepert/mmkb) / [archival record](https://doi.org/10.5281/zenodo.1245698) | Original graph, numerical, image-index, and sameAs resources |
| MMEA | [Repository and dataset instructions](https://github.com/liyichen-cly/MMEA#dataset) | Benchmark context for FB15K-DB15K and FB15K-YAGO15K |
| MEAformer | [Dataset instructions](https://github.com/zjukg/MEAformer#-dataset) / [authors' Google Drive archive](https://drive.google.com/file/d/1VIWcc3KDcLcRImeSrF2AyhetBLq_gsnx/view?usp=sharing) | Alternative prepared MMKG and DBP15K resources |
| EVA | [Data instructions](https://github.com/cambridgeltl/eva#data) | Multi-modal DBP15K/DWY15K features and raw-image links |

The upstream download links above are documented by their respective projects. Cloud-hosted files may require sign-in or have access limits. They do not replace the exact ETA prepared archive: entity mappings, splits, and feature encoders may differ. This release does not implement the three bilingual DBP15K experiments reported in the manuscript.

## Provenance and attribution

The prepared release preserves all data files supplied in `ETA.zip` byte for byte. The data archive adds the upstream MMKB license notice. The MMKB project identifies its terms as BSD-3-Clause; see [MMKB-LICENSE.txt](MMKB-LICENSE.txt) and [ACKNOWLEDGMENTS.md](../ACKNOWLEDGMENTS.md) for attribution and citations. Raw image and pretrained-model terms remain those of their respective sources.

Release preparation validates file hashes, graph endpoints, entity ID separation, pair endpoints, duplicate pairs, train/development disjointness, and complete coverage of reference pairs. It does not rerun feature extraction or training.
