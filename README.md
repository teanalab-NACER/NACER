# NACER: Neural Architecture for Conversational Entity Retrieval

## Overview
NACER (Neural Architecture for Conversational Entity Retrieval) is an advanced feature-based neural architecture designed for the task of KG candidate entity ranking in Conversational Entity Retrieval from DBpedia. This repository contains the source code for NACER and the baseline models utilized in our research, including BM25F, KV-MemNN, GENRE, and LLaMa. Our findings are detailed in the paper: "Benchmark and Neural Architecture for Conversational Entity Retrieval from a Knowledge Graph," presented at the ACM Web Conference 2024 (WWW ’24), authored by Mona Zamiri, Yao Qiang, Fedor Nikolaev, Dongxiao Zhu, Alexander Kotov.

## Dataset
This section provides information on the dataset preparation process, including entity linking and candidate entity collection methods.

### Obsolete Method
- **Entity Linking using TAGME:** For reference, `extract_entities.py` outlines an obsolete method for entity linking.

### Current Method
1. **Mentioned Entities:** Obtain a set of mentioned entities using Lukovnikov's method. Refer to `lukovnikov/README.md` for details.
2. **Candidate Entities:** Collect a set of candidate entities Y with `collect_neighbors.py`.

## Models

### NACER Model
The NACER model is implemented across several files, detailing the process from collecting KG triples to the model's training and testing procedures.
1. `collect_triples.py` - For collecting KG triples Ti.
2. `calculate_feature_inputs.py` - For calculating inputs necessary for semantic similarity features.
3. `calculate_overlap_features.py` - For calculating values for lexical similarity features.
4. `model*_mult*.py` - Contains the model, along with training and testing procedures.

## Baselines
The baseline models are organized in the "Baselines" folder, which includes separate folders for BM25F, KV-MemNN, GENRE, and LLaMa.

- **BM25F:** Implementation details are provided in a separate README file within its folder.
- **KV-MemNN:** Models are implemented in `model_kvmem*.py` and `collect_kvmem_triples.py`.
- **GENRE:** Implementation fdetails are provided in a separate README file within its folder.
- **LLaMa:** Implementation fdetails are provided in a separate README file within its folde.

Statistical significance of the results is calculated using `calculate_stat_significance.py`.

## Citation
If you find this work useful, please cite our paper:
