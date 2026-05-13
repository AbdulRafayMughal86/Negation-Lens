from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from metrics import calculate_all_metrics
from nevir_data import load_nevir_split
from nevir_data import make_prediction_row
from nevir_data import make_query_cases


def score_query_case_with_tfidf(query_case):
    documents = [
        query_case["doc1_text"],
        query_case["doc2_text"],
    ]

    vectorizer = TfidfVectorizer()
    document_vectors = vectorizer.fit_transform(documents)

    query_vector = vectorizer.transform([query_case["query_text"]])
    scores = cosine_similarity(query_vector, document_vectors)[0]

    doc1_score = scores[0]
    doc2_score = scores[1]

    return doc1_score, doc2_score


def make_tfidf_predictions(split_name):
    data = load_nevir_split(split_name)
    query_cases = make_query_cases(data)

    prediction_rows = []

    for query_case in query_cases:
        doc1_score, doc2_score = score_query_case_with_tfidf(query_case)

        prediction_row = make_prediction_row(
            split_name,
            "tfidf",
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

    prediction_rows = make_tfidf_predictions(split_name)
    metrics = calculate_all_metrics(prediction_rows)

    print("model: tfidf")
    print(f"split: {split_name}")
    print(f"query cases: {len(prediction_rows)}")
    print_metrics(metrics)


if __name__ == "__main__":
    main()
