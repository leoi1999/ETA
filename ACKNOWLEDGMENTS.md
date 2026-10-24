# Acknowledgments and third-party resources

## Data provenance

The FB15K-DB15K and FB15K-YAGO15K benchmarks are based on Freebase, DBpedia, and YAGO resources distributed through [MMKG/MMKB](https://github.com/mniepert/mmkb) and used by [MMEA](https://github.com/liyichen-cly/MMEA). ETA's prepared files preserve the entity identifiers, graph resources, splits, metadata, and visual features supplied with the authors' code archive.

MMKB identifies its license as BSD-3-Clause. Its upstream license is reproduced in [docs/MMKB-LICENSE.txt](docs/MMKB-LICENSE.txt) and included in the data archive. This notice does not change the terms applicable to other source resources, raw images, or pretrained models.

```bibtex
@inproceedings{liu2019mmkg,
  title={MMKG: Multi-Modal Knowledge Graphs},
  author={Liu, Ye and Li, Hui and Garcia-Duran, Alberto and Niepert, Mathias and Onoro-Rubio, Daniel and Rosenblum, David S},
  booktitle={European Semantic Web Conference},
  pages={459--474},
  year={2019},
  organization={Springer}
}

@inproceedings{chen2020mmea,
  title={MMEA: Entity Alignment for Multi-modal Knowledge Graph},
  author={Chen, Liyi and Li, Zhi and Wang, Yijun and Xu, Tong and Wang, Zhefeng and Chen, Enhong},
  booktitle={International Conference on Knowledge Science, Engineering and Management},
  pages={134--147},
  year={2020},
  organization={Springer}
}
```

## Models and libraries

The implementation uses PyTorch, PyTorch Scatter, Hugging Face Transformers, NumPy, Pillow, and tqdm. Attribute and optional visual feature extraction use [BERT base uncased](https://huggingface.co/google-bert/bert-base-uncased) and [CLIP ViT-B/16](https://huggingface.co/openai/clip-vit-base-patch16), respectively. Model weights and library source code are obtained from their upstream projects and are not redistributed here.

## Documentation references

The public repositories [MEAformer](https://github.com/zjukg/MEAformer), [EVA](https://github.com/cambridgeltl/eva), and [MMEA](https://github.com/liyichen-cly/MMEA) were consulted for research-repository organization: method overview, framework image, dataset links, dependencies, experiment commands, and citation information. This is a documentation reference, not a claim that ETA's implementation was derived from those repositories.

No software license for ETA was supplied with the original archive. The MMKB license notice above applies to the corresponding upstream resources; it does not license ETA's original code.
