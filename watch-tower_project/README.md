# Stripe Analytics Project

This project restructures the original notebook into a modular Python package with separate layers for:
- data loading
- cleaning
- validation
- feature engineering
- model training
- pipeline orchestration

## Run

```bash
PYTHONPATH=. python src/pipeline.py
```

## Test

```bash
PYTHONPATH=. pytest
```
