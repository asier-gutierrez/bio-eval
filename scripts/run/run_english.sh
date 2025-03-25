#! /bin/zsh

export PYTORCH_ENABLE_MPS_FALLBACK=1

# Discriminative models
python scripts/evaluate/snomed.py \
    --model HiTZ/EriBERTa-base \
    --type MLM \
    --snomed_path ./input/output_corpus_en.jsonl \
    --sampling 0.01 \
    --device mps \
    --output_path ./output/en


python scripts/evaluate/snomed.py \
    --model google-bert/bert-base-multilingual-cased \
    --type MLM \
    --snomed_path ./input/output_corpus_en.jsonl \
    --sampling 0.01 \
    --device mps \
    --output_path ./output/en


python scripts/evaluate/snomed.py \
    --model dmis-lab/biobert-v1.1 \
    --type MLM \
    --snomed_path ./input/output_corpus_en.jsonl \
    --sampling 0.01 \
    --device mps \
    --output_path ./output/en



# Generative models
python scripts/evaluate/snomed.py \
    --model meta-llama/Llama-3.2-3B-Instruct \
    --type CLM \
    --snomed_path ./input/output_corpus_en.jsonl \
    --sampling 0.01 \
    --device mps \
    --output_path ./output/en


python scripts/evaluate/snomed.py \
    --model google/gemma-2-2b-it \
    --type CLM \
    --snomed_path ./input/output_corpus_en.jsonl \
    --sampling 0.01 \
    --device mps \
    --output_path ./output/en
