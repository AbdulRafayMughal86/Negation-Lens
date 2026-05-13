def calculate_query_accuracy(prediction_rows):
    if len(prediction_rows) == 0:
        return 0.0

    correct_count = 0

    for prediction in prediction_rows:
        if prediction["predicted_document_label"] == prediction["correct_document_label"]:
            correct_count = correct_count + 1

    accuracy = correct_count / len(prediction_rows)

    return accuracy


def calculate_pairwise_accuracy(prediction_rows):
    if len(prediction_rows) == 0:
        return 0.0

    predictions_by_row_id = {}

    for prediction in prediction_rows:
        row_id = prediction["row_id"]

        if row_id not in predictions_by_row_id:
            predictions_by_row_id[row_id] = []

        predictions_by_row_id[row_id].append(prediction)

    correct_pair_count = 0
    total_pair_count = 0

    for row_id in predictions_by_row_id:
        row_predictions = predictions_by_row_id[row_id]

        has_q1 = False
        has_q2 = False
        both_queries_are_correct = True

        # A pair is correct only when q1 and q2 are both present and both correct.
        for prediction in row_predictions:
            if prediction["query_label"] == "q1":
                has_q1 = True

            if prediction["query_label"] == "q2":
                has_q2 = True

            if prediction["predicted_document_label"] != prediction["correct_document_label"]:
                both_queries_are_correct = False

        if has_q1 and has_q2:
            total_pair_count = total_pair_count + 1

            if both_queries_are_correct:
                correct_pair_count = correct_pair_count + 1

    if total_pair_count == 0:
        return 0.0

    accuracy = correct_pair_count / total_pair_count

    return accuracy


# Mean Reciprocal Rank Formula for one query: 1 / correct_rank
def calculate_mrr(prediction_rows):
    if len(prediction_rows) == 0:
        return 0.0

    reciprocal_rank_total = 0

    for prediction in prediction_rows:
        reciprocal_rank = 1 / prediction["correct_rank"]
        reciprocal_rank_total = reciprocal_rank_total + reciprocal_rank

    mrr = reciprocal_rank_total / len(prediction_rows)

    return mrr


def calculate_all_metrics(prediction_rows):
    metrics = {
        "query_accuracy": calculate_query_accuracy(prediction_rows),
        "pairwise_accuracy": calculate_pairwise_accuracy(prediction_rows),
        "mrr": calculate_mrr(prediction_rows),
    }

    return metrics
