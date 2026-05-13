import random

from metrics import calculate_all_metrics
from nevir_data import load_nevir_split
from nevir_data import make_prediction_row
from nevir_data import make_query_cases


RANDOM_SEED = random.randint(0, 1000000)


def make_random_predictions(split_name):
    data = load_nevir_split(split_name)
    query_cases = make_query_cases(data)

    random_number_generator = random.Random(RANDOM_SEED)
    prediction_rows = []

    for query_case in query_cases:
        doc1_score = random_number_generator.random()
        doc2_score = random_number_generator.random()

        prediction_row = make_prediction_row(
            split_name,
            "random",
            query_case,
            doc1_score,
            doc2_score,
        )

        prediction_rows.append(prediction_row)

    return prediction_rows


def print_metrics(metrics):
    print(f"query_accuracy: {metrics['query_accuracy']:.3f}")
    print(f"pairwise_accuracy: {metrics['pairwise_accuracy']:.3f}")
    print(f"mrr: {metrics['mrr']:.3f}")


def main():
    split_name = "validation"

    prediction_rows = make_random_predictions(split_name)
    metrics = calculate_all_metrics(prediction_rows)

    print("model: random")
    print(f"split: {split_name}")
    print(f"query cases: {len(prediction_rows)}")
    print_metrics(metrics)


if __name__ == "__main__":
    main()
