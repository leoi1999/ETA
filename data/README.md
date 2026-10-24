# Prepared data

From the repository root, run:

```bash
python tools/download_data.py
```

Alternatively, download [eta-data-v1.zip](https://github.com/leoi1999/ETA/releases/download/data-v1/eta-data-v1.zip) and extract it in the repository root. The archive supplies `data/FB15K_DB15K/` and `data/FB15K_YAGO15K/`.

For an archive already downloaded to disk:

```bash
python tools/download_data.py --archive /path/to/eta-data-v1.zip
```

To verify existing files without downloading anything:

```bash
python tools/download_data.py --check
```

See [dataset documentation](../docs/DATASETS.md) for file formats, provenance, and statistics. Dataset files are distributed as a release asset and excluded from Git history.
