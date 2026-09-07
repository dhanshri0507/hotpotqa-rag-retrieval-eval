#!/usr/bin/env bash
set -e

python build_dataset.py
python -m src.bm25_retrieval
python -m src.dense_retrieval
python -m src.hybrid_retrieval
python -m src.evaluate
python -m src.failure_analysis
python -m src.improved_retrieval
python -m src.evaluate_improvement
