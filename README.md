# Negation-Lens

Negation-Lens is a simple information retrieval project for testing how well retrieval models handle negation in queries.

The project uses the NevIR dataset, where each row contains two related queries and two related documents:

```text
q1 should retrieve doc1
q2 should retrieve doc2
```

This makes it possible to test whether a model understands meaning changes caused by words like `not`, `without`, `cannot`, or `less than`.

## Current Models

The current pipeline evaluates:

- Random baseline
- BM25
- TF-IDF

Sentence-BERT is planned for a later phase.

## Dataset

The dataset files are already saved locally:

```text
data/train.csv
data/validation.csv
data/test.csv
```

All three files use the same columns:

```text
id, WorkerId, q1, q2, doc1, doc2
```

Current development runs use `validation`. The `test` split should be kept for final evaluation.

## Setup

Create and activate a virtual environment if needed:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install requirements:

```bash
.venv/bin/python -m pip install -r requirements.txt
```

## Run Evaluation

Run all implemented models on validation:

```bash
.venv/bin/python src/evaluate.py --split validation --model all
```

Run one model:

```bash
.venv/bin/python src/evaluate.py --split validation --model bm25
```

Supported model options:

```text
random, bm25, tfidf, all
```

Supported split options:

```text
validation, test
```

## Current Validation Results

```text
model   query_acc   pairwise_acc   mrr
random  0.509       0.249          0.754
bm25    0.491       0.022          0.746
tfidf   0.502       0.062          0.751
```

`query_acc` checks each query separately.

`pairwise_acc` is stricter. A row is correct only if both `q1` and `q2` are correct.

`mrr` means Mean Reciprocal Rank. Higher is better.

## Saved Outputs

Running `src/evaluate.py` saves:

```text
results/predictions.csv
results/model_metrics.csv
```

`predictions.csv` stores one row per model prediction.

`model_metrics.csv` stores the summary scores for each model.

The `results/` folder is ignored by git because these files can be regenerated.

## Main Files

```text
src/nevir_data.py        Load data and create query cases
src/metrics.py           Calculate query accuracy, pairwise accuracy, and MRR
src/random_baseline.py   Random baseline
src/bm25_baseline.py     BM25 baseline
src/tfidf_baseline.py    TF-IDF baseline
src/evaluate.py          Main evaluation runner
```
