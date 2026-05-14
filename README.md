# Negation-Lens

Negation-Lens is an information retrieval project that tests how well common retrieval models understand negation.

The project uses the NevIR dataset, where each example contains two related queries and two related documents. The task is simple to state:

```text
q1 should retrieve doc1
q2 should retrieve doc2
```

The paired queries often differ by negation or a meaning reversal, such as `not`, `without`, `cannot`, `less than`, or antonym-style wording. This makes the dataset useful for checking whether a retrieval model is matching meaning or only matching surface words.

## What This Repo Includes

- NevIR train, validation, and test CSV files
- Random, BM25, TF-IDF, and Sentence-BERT retrieval baselines
- Query-level accuracy, pairwise accuracy, and MRR evaluation
- Rule-based negation type labels: `explicit`, `implicit`, `comparative`, and `other`
- Final result tables, plots, and selected error examples

## Models

The current pipeline evaluates:

```text
random
bm25
tfidf
sbert
```

`sbert` uses Sentence-BERT with `sentence-transformers/all-MiniLM-L6-v2`.

## Dataset

Dataset source: [orionweller/NevIR on Hugging Face](https://huggingface.co/datasets/orionweller/NevIR)

The dataset splits are stored locally:

```text
data/train.csv
data/validation.csv
data/test.csv
```

Each CSV uses this structure:

```text
id, WorkerId, q1, q2, doc1, doc2
```

For every row, `q1` should match `doc1`, and `q2` should match `doc2`.

## Setup

Create a virtual environment:

```bash
python3 -m venv .venv
```

Activate it on Linux/macOS:

```bash
source .venv/bin/activate
```

Activate it on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install the standard dependencies:

```bash
python -m pip install -r config/requirements.txt
```

To run `sbert` or `all`, install the dense retrieval dependency too:

```bash
python -m pip install -r config/requirements-dense.txt
```

Sentence-BERT downloads its model from Hugging Face the first time it runs. If you do not want that download, run only `random`, `bm25`, or `tfidf`.

## Run Evaluation

Run all models on validation:

```bash
python src/evaluate.py --split validation --model all
```

Run all models on test:

```bash
python src/evaluate.py --split test --model all
```

Run a single model:

```bash
python src/evaluate.py --split validation --model bm25
```

Supported splits:

```text
validation, test
```

Supported models:

```text
random, bm25, tfidf, sbert, all
```

## Build Report Files

After running evaluation, build the final report outputs:

```bash
python src/report.py
```

This creates final comparison tables, plots, and selected error examples in `results/`.

## Results

### Validation

```text
model   query_acc   pairwise_acc   mrr     query_cases
random  0.509       0.249          0.754   450
bm25    0.491       0.022          0.746   450
tfidf   0.502       0.062          0.751   450
sbert   0.522       0.084          0.761   450
```

### Test

```text
model   query_acc   pairwise_acc   mrr     query_cases
random  0.492       0.244          0.746   2766
bm25    0.484       0.018          0.742   2766
tfidf   0.484       0.043          0.742   2766
sbert   0.523       0.071          0.762   2766
```

Sentence-BERT gives the best query accuracy and MRR on both validation and test. Pairwise accuracy stays low across the learned and lexical baselines, showing that paired negation queries need more careful retrieval methods than basic similarity alone.

## Metrics

`query_acc` checks each query independently.

`pairwise_acc` is stricter. A dataset row is counted as correct only if both `q1` and `q2` retrieve the right document.

`mrr` means Mean Reciprocal Rank. Higher is better.

## Negation Type Labels

The project also labels queries with simple rule-based categories:

```text
explicit      Direct negation words such as not, no, never, cannot, or n't
implicit      Absence-style words such as without, except, lack, or lacking
comparative   Comparison phrases such as less than, more than, at least, or at most
other         Queries that do not match the simple keyword rules
```

These labels are used for per-type result breakdowns.

## Output Files

Evaluation outputs:

```text
results/validation_predictions.csv
results/validation_model_metrics.csv
results/validation_negation_type_metrics.csv
results/test_predictions.csv
results/test_model_metrics.csv
results/test_negation_type_metrics.csv
```

Final report outputs:

```text
results/final_model_comparison.csv
results/final_per_type_comparison.csv
results/error_examples.csv
results/plots/
```

## Plots

Validation:

![Validation MRR](results/plots/validation_mrr.png)

![Validation pairwise accuracy](results/plots/validation_pairwise_accuracy.png)

![Validation negation type accuracy](results/plots/validation_negation_type_accuracy.png)

Test:

![Test MRR](results/plots/test_mrr.png)

![Test pairwise accuracy](results/plots/test_pairwise_accuracy.png)

![Test negation type accuracy](results/plots/test_negation_type_accuracy.png)

## Project Structure

```text
config/
  requirements.txt           Standard dependencies
  requirements-dense.txt     Sentence-BERT dependency

data/
  train.csv
  validation.csv
  test.csv

results/
  Final CSV outputs and plots

scripts/
  download_nevir_data.py     Dataset download helper

src/
  nevir_data.py              Load data and create query cases
  metrics.py                 Calculate query accuracy, pairwise accuracy, and MRR
  negation_labeler.py        Label queries by simple negation type rules
  random_baseline.py         Random baseline
  bm25_baseline.py           BM25 baseline
  tfidf_baseline.py          TF-IDF baseline
  sbert_baseline.py          Sentence-BERT baseline
  evaluate.py                Main evaluation runner
  report.py                  Build final tables, plots, and error examples
```

---

<h2 align="center">Thanks for Checking This Out 👋</h2>

<p align="center"><em>Read the query carefully. Track the negation. Retrieve the meaning, not just the matching words.</em></p>
