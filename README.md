# ssqa-kg
codes for paper named "Benchmark and Neural Architecture for Conversational Entity Retrieval from a Knowledge Graph"

# Dataset
Obsolete: Entity linking using TAGME: `extract_entities.py`
1. Obtaining a set of mentioned entities using Lukovnikov's method `lukovnikov/README.md`
2. Obtaining a set of candidate entities Y: `collect_neighbors.py`

# Models

NASS-QA model is implemented in the following files:
  1. Collecting KG triples Ti: `collect_triples.py`
  2. Calculating inputs for semantic similarity features: `calculate_feature_inputs.py`
  3. Calculating values for lexical similarity features: `calculate_overlap_features.py`
  4. The model, training and testing procedures: `model*_mult*.py`

KV-MemNN models are implemented in files `model_kvmem*.py` and `collect_kvmem_triples.py`

Statistical significance is calculated in `calculate_stat_significance.py`
