# ZnBERT: Domain-Adaptive BERT for Zinc Alloy Literature
ZnBERT is a domain-adapted language model tailored for zinc alloy research. Starting from a general-purpose BERT backbone, we perform continued masked-language-model (MLM) pretraining on a curated corpus of Zn-alloy abstracts. The goal is to learn Zn-specific terminology and composition/processing expressions (e.g., alloying elements, deformation routes, temperatures, reductions, ECAP passes), enabling stronger text representations for downstream tasks such as predicting mechanical properties (UTS/YS/EL), extracting composition–processing descriptors, and supporting data-driven alloy design.


ZnBERT/
  ├─ ZnBERTv2_fromBERT_DAPT_Abstract/
  │   ├─ config.json
  │   ├─ model.safetensors
  │   ├─ tokenizer.json
  │   ├─ tokenizer_config.json
  │   ├─ special_tokens_map.json
  │   └─ vocab.txt
  ├─ scripts/
  │   ├─ train_mlm.py
  │   └─ inference_embedding.py
  ├─ README.md
  ├─ LICENSE
  └─ requirements.txt
