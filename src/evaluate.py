import argparse
import csv
from pathlib import Path

from bm25_baseline import make_bm25_predictions
from metrics import calculate_all_metrics
from random_baseline import make_random_predictions
from tfidf_baseline import make_tfidf_predictions


MODEL_NAMES = ["random", "bm25", "tfidf"]
RESULTS_FOLDER = Path("results")
PREDICTIONS_FILE = RESULTS_FOLDER / "predictions.csv"
METRICS_FILE = RESULTS_FOLDER / "model_metrics.csv"


def get_prediction_rows(model_name, split_name):
    if model_name == "random":
        return make_random_predictions(split_name)

    if model_name == "bm25":
        return make_bm25_predictions(split_name)

    if model_name == "tfidf":
        return make_tfidf_predictions(split_name)

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


def run_evaluation(split_name, model_name):
    models_to_run = get_models_to_run(model_name)
    results = []

    for current_model_name in models_to_run:
        prediction_rows = get_prediction_rows(current_model_name, split_name)
        metrics = calculate_all_metrics(prediction_rows)

        result = {
            "model": current_model_name,
            "metrics": metrics,
            "query_cases": len(prediction_rows),
            "prediction_rows": prediction_rows,
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


def save_results(split_name, results):
    prediction_rows = collect_prediction_rows(results)
    metric_rows = make_metric_rows(split_name, results)

    save_rows_to_csv(PREDICTIONS_FILE, prediction_rows)
    save_rows_to_csv(METRICS_FILE, metric_rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["validation", "test"], default="validation")
    parser.add_argument("--model", choices=["random", "bm25", "tfidf", "all"], default="all")
    args = parser.parse_args()

    results = run_evaluation(args.split, args.model)

    print(f"split: {args.split}")
    print_metrics_table(results)

    save_results(args.split, results)
    print(f"saved predictions: {PREDICTIONS_FILE}")
    print(f"saved metrics: {METRICS_FILE}")


if __name__ == "__main__":
    main()
