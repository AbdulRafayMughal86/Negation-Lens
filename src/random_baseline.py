import random

from nevir_data import load_nevir_split
from nevir_data import make_prediction_row
from nevir_data import make_query_cases


RANDOM_SEED = 42


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
