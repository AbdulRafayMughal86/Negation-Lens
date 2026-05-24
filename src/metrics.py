# calculate metrics for the predictions
def calculate_query_accuracy(prediction_rows):
    if len(prediction_rows) == 0:
        return 0.0

    correct_count = 0

    for prediction in prediction_rows:
        if prediction["predicted_document_label"] == prediction["correct_document_label"]:
            correct_count = correct_count + 1

    accuracy = correct_count / len(prediction_rows)

    return accuracy

# A pair is correct only when q1 and q2 are both present and both correct.
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

    # for each row_id, check if both q1 and q2 are present and correct
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

# Get MonoT5 scores for a list of (query, document) pairs
def calculate_all_metrics(prediction_rows):
    metrics = {
        "query_accuracy": calculate_query_accuracy(prediction_rows),
        "pairwise_accuracy": calculate_pairwise_accuracy(prediction_rows),
        "mrr": calculate_mrr(prediction_rows),
    }

    return metrics

# Calculate metrics for each negation type
def calculate_per_type_metrics(prediction_rows):
    rows_by_type = {}

    for prediction in prediction_rows:
        negation_type = prediction.get("negation_type", "other")
        if negation_type not in rows_by_type:
            rows_by_type[negation_type] = []
        rows_by_type[negation_type].append(prediction)

    per_type_rows = []

    for negation_type, type_rows in rows_by_type.items():
        per_type_rows.append({
            "negation_type": negation_type,
            "query_accuracy": calculate_query_accuracy(type_rows),
            "mrr": calculate_mrr(type_rows),
            "query_cases": len(type_rows),
        })

    per_type_rows.sort(key=lambda r: r["negation_type"])

    return per_type_rows
