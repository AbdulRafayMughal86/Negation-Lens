from pathlib import Path

import pandas as pd

from negation_labeler import label_negation_type


DATA_FOLDER = Path("data")

VALID_SPLITS = ["train", "validation", "test"]

REQUIRED_COLUMNS = [
    "id",
    "WorkerId",
    "q1",
    "q2",
    "doc1",
    "doc2",
]


def load_nevir_split(split_name):
    if split_name not in VALID_SPLITS:
        valid_split_text = ", ".join(VALID_SPLITS)
        error_message = f"Invalid split '{split_name}'. Use one of: {valid_split_text}."
        raise ValueError(error_message)

    csv_file_path = DATA_FOLDER / f"{split_name}.csv"

    if not csv_file_path.exists():
        error_message = f"Could not find this data file: {csv_file_path}"
        raise FileNotFoundError(error_message)

    data = pd.read_csv(csv_file_path)

    missing_columns = []

    # Check every required column one by one so the error is easy to understand.
    for column_name in REQUIRED_COLUMNS:
        if column_name not in data.columns:
            missing_columns.append(column_name)

    if missing_columns:
        missing_column_text = ", ".join(missing_columns)
        error_message = f"{csv_file_path} is missing these columns: {missing_column_text}"
        raise ValueError(error_message)

    return data


def make_query_cases(data):
    query_cases = []

    for row_number in range(len(data)):
        row = data.iloc[row_number]

        # First query should match the first document.
        q1_case = {
            "row_id": row["id"],
            "query_label": "q1",
            "query_text": row["q1"],
            "correct_document_label": "doc1",
            "correct_document_text": row["doc1"],
            "doc1_text": row["doc1"],
            "doc2_text": row["doc2"],
        }

        # Second query should match the second document.
        q2_case = {
            "row_id": row["id"],
            "query_label": "q2",
            "query_text": row["q2"],
            "correct_document_label": "doc2",
            "correct_document_text": row["doc2"],
            "doc1_text": row["doc1"],
            "doc2_text": row["doc2"],
        }

        query_cases.append(q1_case)
        query_cases.append(q2_case)

    return query_cases


def make_prediction_row(split_name, model_name, query_case, doc1_score, doc2_score):
    if doc1_score >= doc2_score:
        predicted_document_label = "doc1"
    else:
        predicted_document_label = "doc2"

    if predicted_document_label == query_case["correct_document_label"]:
        correct_rank = 1
    else:
        correct_rank = 2

    negation_type = label_negation_type(query_case["query_text"])

    prediction_row = {
        "split": split_name,
        "model": model_name,
        "row_id": query_case["row_id"],
        "query_label": query_case["query_label"],
        "query_text": query_case["query_text"],
        "negation_type": negation_type,
        "correct_document_label": query_case["correct_document_label"],
        "predicted_document_label": predicted_document_label,
        "correct_rank": correct_rank,
        "doc1_score": doc1_score,
        "doc2_score": doc2_score,
    }

    return prediction_row
