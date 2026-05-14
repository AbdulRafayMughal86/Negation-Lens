import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RESULTS_FOLDER = Path("results")
PLOTS_FOLDER = RESULTS_FOLDER / "plots"
PLOT_COLORS = ["steelblue", "coral", "forestgreen", "mediumpurple", "goldenrod"]


def _read_csv(file_path):
    with file_path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _write_csv(file_path, rows):
    if not rows:
        return
    with file_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def load_all_model_metrics():
    rows = []
    for path in [RESULTS_FOLDER / "validation_model_metrics.csv", RESULTS_FOLDER / "test_model_metrics.csv"]:
        if not path.exists():
            continue
        rows.extend(_read_csv(path))
    return rows


def load_all_per_type_metrics():
    rows = []
    for path in [RESULTS_FOLDER / "validation_negation_type_metrics.csv", RESULTS_FOLDER / "test_negation_type_metrics.csv"]:
        if not path.exists():
            continue
        rows.extend(_read_csv(path))
    return rows


def load_all_predictions():
    rows = []
    for path in [RESULTS_FOLDER / "validation_predictions.csv", RESULTS_FOLDER / "test_predictions.csv"]:
        if not path.exists():
            continue
        rows.extend(_read_csv(path))
    return rows


# ── Phase 16: Final Result Tables ────────────────────────────────────────────

def build_model_comparison_table(raw_rows):
    result = []
    for r in raw_rows:
        result.append({
            "split": r["split"],
            "model": r["model"],
            "query_accuracy": float(r["query_accuracy"]),
            "pairwise_accuracy": float(r["pairwise_accuracy"]),
            "mrr": float(r["mrr"]),
            "pair_count": int(r["query_cases"]) // 2,
            "query_cases": int(r["query_cases"]),
        })
    return result


def build_per_type_table(raw_rows):
    result = []
    for r in raw_rows:
        result.append({
            "split": r["split"],
            "model": r["model"],
            "negation_type": r["negation_type"],
            "query_accuracy": float(r["query_accuracy"]),
            "mrr": float(r["mrr"]),
            "query_cases": int(r["query_cases"]),
        })
    return result


def save_final_tables(model_rows, per_type_rows):
    RESULTS_FOLDER.mkdir(exist_ok=True)
    model_path = RESULTS_FOLDER / "final_model_comparison.csv"
    per_type_path = RESULTS_FOLDER / "final_per_type_comparison.csv"
    _write_csv(model_path, model_rows)
    _write_csv(per_type_path, per_type_rows)
    return model_path, per_type_path


# ── Phase 17: Plots ───────────────────────────────────────────────────────────

def make_plots(model_rows, per_type_rows):
    PLOTS_FOLDER.mkdir(parents=True, exist_ok=True)
    splits = sorted(set(r["split"] for r in model_rows))

    for split in splits:
        split_model_rows = [r for r in model_rows if r["split"] == split]
        split_per_type_rows = [r for r in per_type_rows if r["split"] == split]
        _plot_bar(split, split_model_rows, "pairwise_accuracy", "Pairwise Accuracy", "steelblue")
        _plot_bar(split, split_model_rows, "mrr", "MRR", "coral")
        _plot_negation_type_accuracy(split, split_per_type_rows)


def _plot_bar(split, rows, metric_key, ylabel, color):
    models = [r["model"] for r in rows]
    values = [r[metric_key] for r in rows]

    fig, ax = plt.subplots(figsize=(9, 4))
    bars = ax.bar(models, values, color=color, width=0.5)
    ax.set_title(f"{ylabel} by Model ({split})")
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Model")
    ax.set_ylim(0, 1)
    ax.tick_params(axis="x", labelrotation=20)
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{val:.3f}",
            ha="center",
            fontsize=9,
        )
    plt.tight_layout()
    plt.savefig(PLOTS_FOLDER / f"{split}_{metric_key}.png", dpi=100)
    plt.close()


def _plot_negation_type_accuracy(split, rows):
    models = sorted(set(r["model"] for r in rows))
    neg_types = sorted(set(r["negation_type"] for r in rows))

    x = np.arange(len(neg_types))
    width = 0.8 / max(len(models), 1)

    fig, ax = plt.subplots(figsize=(11, 5))

    for i, model in enumerate(models):
        model_by_type = {r["negation_type"]: r["query_accuracy"] for r in rows if r["model"] == model}
        values = [model_by_type.get(t, 0.0) for t in neg_types]
        offset = (i - len(models) / 2 + 0.5) * width
        ax.bar(x + offset, values, width=width, label=model, color=PLOT_COLORS[i % len(PLOT_COLORS)])

    ax.set_title(f"Query Accuracy by Negation Type ({split})")
    ax.set_ylabel("Query Accuracy")
    ax.set_xlabel("Negation Type")
    ax.set_xticks(x)
    ax.set_xticklabels(neg_types)
    ax.set_ylim(0, 1)
    ax.legend()
    plt.tight_layout()
    plt.savefig(PLOTS_FOLDER / f"{split}_negation_type_accuracy.png", dpi=100)
    plt.close()


# ── Phase 18: Error Examples ──────────────────────────────────────────────────

def extract_error_examples(prediction_rows, max_examples=100):
    non_random = [r for r in prediction_rows if r["model"] != "random"]

    by_query = {}
    for row in non_random:
        key = (row["split"], row["row_id"], row["query_label"])
        if key not in by_query:
            by_query[key] = []
        by_query[key].append(row)

    error_rows = []

    for _key, preds in by_query.items():
        wrong = [p for p in preds if p["predicted_document_label"] != p["correct_document_label"]]
        right = [p for p in preds if p["predicted_document_label"] == p["correct_document_label"]]

        if not wrong:
            continue

        is_disagreement = bool(right) and bool(wrong)

        for pred in wrong:
            error_rows.append({
                "split": pred["split"],
                "row_id": pred["row_id"],
                "query_label": pred["query_label"],
                "query_text": pred["query_text"],
                "correct_document_label": pred["correct_document_label"],
                "predicted_document_label": pred["predicted_document_label"],
                "model": pred["model"],
                "negation_type": pred["negation_type"],
                "_priority": 0 if is_disagreement else 1,
            })

    error_rows.sort(key=lambda r: (r["_priority"], r["row_id"]))

    for r in error_rows:
        del r["_priority"]

    return error_rows[:max_examples]


def save_error_examples(error_rows):
    RESULTS_FOLDER.mkdir(exist_ok=True)
    path = RESULTS_FOLDER / "error_examples.csv"
    _write_csv(path, error_rows)
    return path


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("loading saved results...")
    raw_model_rows = load_all_model_metrics()
    raw_per_type_rows = load_all_per_type_metrics()
    prediction_rows = load_all_predictions()
    print(f"  {len(raw_model_rows)} model metric rows")
    print(f"  {len(raw_per_type_rows)} per-type metric rows")
    print(f"  {len(prediction_rows)} prediction rows")

    print("\nphase 16: building final report tables...")
    model_rows = build_model_comparison_table(raw_model_rows)
    per_type_rows = build_per_type_table(raw_per_type_rows)
    model_path, per_type_path = save_final_tables(model_rows, per_type_rows)
    print(f"  saved: {model_path}")
    print(f"  saved: {per_type_path}")

    print("\nphase 17: generating plots...")
    make_plots(model_rows, per_type_rows)
    for plot_file in sorted(PLOTS_FOLDER.glob("*.png")):
        print(f"  saved: {plot_file}")

    print("\nphase 18: extracting error examples...")
    error_rows = extract_error_examples(prediction_rows)
    error_path = save_error_examples(error_rows)
    print(f"  saved: {error_path} ({len(error_rows)} rows)")


if __name__ == "__main__":
    main()
