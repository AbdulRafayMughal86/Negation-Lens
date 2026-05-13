import numpy as np
from sentence_transformers import SentenceTransformer

from metrics import calculate_all_metrics
from nevir_data import load_nevir_split
from nevir_data import make_prediction_row
from nevir_data import make_query_cases


MODEL_ID = "all-MiniLM-L6-v2"


def cosine_similarity(vec_a, vec_b):
    dot = np.dot(vec_a, vec_b)
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def make_sbert_predictions(split_name):
    data = load_nevir_split(split_name)
    query_cases = make_query_cases(data)

    model = SentenceTransformer(MODEL_ID)

    query_texts = [case["query_text"] for case in query_cases]
    doc1_texts = [case["doc1_text"] for case in query_cases]
    doc2_texts = [case["doc2_text"] for case in query_cases]

    all_texts = query_texts + doc1_texts + doc2_texts
    all_embeddings = model.encode(all_texts, batch_size=64, show_progress_bar=False)

    n = len(query_cases)
    query_embeddings = all_embeddings[:n]
    doc1_embeddings = all_embeddings[n : n * 2]
    doc2_embeddings = all_embeddings[n * 2 :]

    prediction_rows = []

    for i, query_case in enumerate(query_cases):
        doc1_score = float(cosine_similarity(query_embeddings[i], doc1_embeddings[i]))
        doc2_score = float(cosine_similarity(query_embeddings[i], doc2_embeddings[i]))

        prediction_row = make_prediction_row(
            split_name,
            "sbert",
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

    prediction_rows = make_sbert_predictions(split_name)
    metrics = calculate_all_metrics(prediction_rows)

    print("model: sbert")
    print(f"split: {split_name}")
    print(f"query cases: {len(prediction_rows)}")
    print_metrics(metrics)


if __name__ == "__main__":
    main()
