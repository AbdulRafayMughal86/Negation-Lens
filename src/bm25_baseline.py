from rank_bm25 import BM25Okapi

from nevir_data import load_nevir_split
from nevir_data import make_prediction_row
from nevir_data import make_query_cases


def tokenize_text(text):
    lowercase_text = text.lower()
    words = lowercase_text.split()

    return words


def score_query_case_with_bm25(query_case):
    document_tokens = [
        tokenize_text(query_case["doc1_text"]),
        tokenize_text(query_case["doc2_text"]),
    ]

    bm25_model = BM25Okapi(document_tokens)

    query_tokens = tokenize_text(query_case["query_text"])
    scores = bm25_model.get_scores(query_tokens)

    doc1_score = scores[0]
    doc2_score = scores[1]

    return doc1_score, doc2_score


def make_bm25_predictions(split_name):
    data = load_nevir_split(split_name)
    query_cases = make_query_cases(data)

    prediction_rows = []

    for query_case in query_cases:
        doc1_score, doc2_score = score_query_case_with_bm25(query_case)

        prediction_row = make_prediction_row(
            split_name,
            "bm25",
            query_case,
            doc1_score,
            doc2_score,
        )

        prediction_rows.append(prediction_row)

    return prediction_rows
