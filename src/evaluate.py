import argparse
import csv
from pathlib import Path

from bm25_baseline import make_bm25_predictions
from metrics import calculate_all_metrics
from metrics import calculate_per_type_metrics
from random_baseline import make_random_predictions
from tfidf_baseline import make_tfidf_predictions


MODEL_NAMES = ["random", "bm25", "tfidf", "sbert"]
RESULTS_FOLDER = Path("results")


def get_result_paths(split_name):
    return {
        "predictions": RESULTS_FOLDER / f"{split_name}_predictions.csv",
        "metrics": RESULTS_FOLDER / f"{split_name}_model_metrics.csv",
        "negation_type_metrics": RESULTS_FOLDER / f"{split_name}_negation_type_metrics.csv",
    }


def get_prediction_rows(model_name, split_name):
    if model_name == "random":
        return make_random_predictions(split_name)

    if model_name == "bm25":
        return make_bm25_predictions(split_name)

    if model_name == "tfidf":
        return make_tfidf_predictions(split_name)

    if model_name == "sbert":
        from sbert_baseline import make_sbert_predictions
        return make_sbert_predictions(split_name)

    raise ValueError(f"Unknown model: {model_name}")


def get_models_to_run(model_name):
    if model_name == "all":
        return MODEL_NAMES

    return [model_name]


def print_metrics_table(results):
    print("model   query_acc   pairwise_acc   mrr     query_cases")
    print("-----   ---------   ------------   -----   -----------")

    for result in results:
        model_name = result["model"]
        metrics = result["metrics"]
        query_cases = result["query_cases"]

        print(
            f"{model_name:<7} "
            f"{metrics['query_accuracy']:.3f}       "
            f"{metrics['pairwise_accuracy']:.3f}          "
            f"{metrics['mrr']:.3f}   "
            f"{query_cases}"
        )


def print_negation_type_table(split_name, results):
    print("\nnegation type breakdown:")
    print(f"{'model':<7}  {'type':<12}  {'query_acc':<10}  {'mrr':<6}  {'cases'}")
    print("-" * 55)

    for result in results:
        model_name = result["model"]
        for row in result["per_type_metrics"]:
            print(
                f"{model_name:<7}  "
                f"{row['negation_type']:<12}  "
                f"{row['query_accuracy']:.3f}       "
                f"{row['mrr']:.3f}  "
                f"{row['query_cases']}"
            )


def run_evaluation(split_name, model_name):
    models_to_run = get_models_to_run(model_name)
    results = []

    for current_model_name in models_to_run:
        prediction_rows = get_prediction_rows(current_model_name, split_name)
        metrics = calculate_all_metrics(prediction_rows)
        per_type_metrics = calculate_per_type_metrics(prediction_rows)

        result = {
            "model": current_model_name,
            "metrics": metrics,
            "query_cases": len(prediction_rows),
            "prediction_rows": prediction_rows,
            "per_type_metrics": per_type_metrics,
        }

        results.append(result)

    return results


def save_rows_to_csv(file_path, rows):
    if len(rows) == 0:
        return

    RESULTS_FOLDER.mkdir(exist_ok=True)

    column_names = list(rows[0].keys())

    with file_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=column_names)
        writer.writeheader()
        writer.writerows(rows)


def collect_prediction_rows(results):
    all_prediction_rows = []

    for result in results:
        for prediction_row in result["prediction_rows"]:
            all_prediction_rows.append(prediction_row)

    return all_prediction_rows


def make_metric_rows(split_name, results):
    metric_rows = []

    for result in results:
        metrics = result["metrics"]

        metric_row = {
            "split": split_name,
            "model": result["model"],
            "query_accuracy": metrics["query_accuracy"],
            "pairwise_accuracy": metrics["pairwise_accuracy"],
            "mrr": metrics["mrr"],
            "query_cases": result["query_cases"],
        }

        metric_rows.append(metric_row)

    return metric_rows


def make_negation_type_metric_rows(split_name, results):
    all_rows = []

    for result in results:
        model_name = result["model"]

        for per_type_row in result["per_type_metrics"]:
            row = {
                "split": split_name,
                "model": model_name,
                "negation_type": per_type_row["negation_type"],
                "query_accuracy": per_type_row["query_accuracy"],
                "mrr": per_type_row["mrr"],
                "query_cases": per_type_row["query_cases"],
            }
            all_rows.append(row)

    return all_rows


def save_results(split_name, results):
    paths = get_result_paths(split_name)

    prediction_rows = collect_prediction_rows(results)
    metric_rows = make_metric_rows(split_name, results)
    negation_type_rows = make_negation_type_metric_rows(split_name, results)

    save_rows_to_csv(paths["predictions"], prediction_rows)
    save_rows_to_csv(paths["metrics"], metric_rows)
    save_rows_to_csv(paths["negation_type_metrics"], negation_type_rows)

    return paths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["validation", "test"], default="validation")
    parser.add_argument("--model", choices=["random", "bm25", "tfidf", "sbert", "all"], default="all")
    args = parser.parse_args()

    results = run_evaluation(args.split, args.model)

    print(f"split: {args.split}")
    print_metrics_table(results)
    print_negation_type_table(args.split, results)

    paths = save_results(args.split, results)
    print(f"\nsaved predictions:           {paths['predictions']}")
    print(f"saved metrics:               {paths['metrics']}")
    print(f"saved negation type metrics: {paths['negation_type_metrics']}")


if __name__ == "__main__":
    main()
