#! /bin/zsh

export PYTORCH_ENABLE_MPS_FALLBACK=1

# Discriminative models
python scripts/evaluate/snomed.py \
    --model HiTZ/EriBERTa-base \
    --type MLM \
    --snomed_path ./input/output_corpus_es.jsonl \
    --sampling 0.01 \
    --device mps \
    --output_path ./output/es


python scripts/evaluate/snomed.py \
    --model google-bert/bert-base-multilingual-cased \
    --type MLM \
    --snomed_path ./input/output_corpus_es.jsonl \
    --sampling 0.01 \
    --device mps \
    --output_path ./output/es


python scripts/evaluate/snomed.py \
    --model PlanTL-GOB-ES/bsc-bio-ehr-es \
    --type MLM \
    --snomed_path ./input/output_corpus_es.jsonl \
    --sampling 0.01 \
    --device mps \
    --output_path ./output/es



# Generative models
python scripts/evaluate/snomed.py \
    --model meta-llama/Llama-3.2-3B-Instruct \
    --type CLM \
    --snomed_path ./input/output_corpus_es.jsonl \
    --sampling 0.01 \
    --device mps \
    --output_path ./output/es


python scripts/evaluate/snomed.py \
    --model google/gemma-2-2b-it \
    --type CLM \
    --snomed_path ./input/output_corpus_es.jsonl \
    --sampling 0.01 \
    --device mps \
    --output_path ./output/es
