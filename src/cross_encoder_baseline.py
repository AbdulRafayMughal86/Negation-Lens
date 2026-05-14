import torch
from transformers import AutoModelForSeq2SeqLM
from transformers import AutoTokenizer

from nevir_data import load_nevir_split
from nevir_data import make_prediction_row
from nevir_data import make_query_cases


MODEL_ID = "castorini/monot5-3b-msmarco-10k"
MODEL_NAME = "monot5_3b"


def make_monot5_scores(text_pairs, model_id, batch_size):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_dtype = torch.float16 if device == "cuda" else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id, torch_dtype=model_dtype)
    model.to(device)
    model.eval()

    true_token_id = tokenizer.encode("true", add_special_tokens=False)[0]
    false_token_id = tokenizer.encode("false", add_special_tokens=False)[0]

    scores = []

    for start in range(0, len(text_pairs), batch_size):
        batch_pairs = text_pairs[start : start + batch_size]
        batch_prompts = []

        for query_text, document_text in batch_pairs:
            prompt = f"Query: {query_text} Document: {document_text} Relevant:"
            batch_prompts.append(prompt)

        inputs = tokenizer(
            batch_prompts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        inputs = {key: value.to(device) for key, value in inputs.items()}

        decoder_input_ids = torch.full(
            (len(batch_prompts), 1),
            model.config.decoder_start_token_id,
            dtype=torch.long,
            device=device,
        )

        with torch.no_grad():
            outputs = model(**inputs, decoder_input_ids=decoder_input_ids)
            logits = outputs.logits[:, 0, [true_token_id, false_token_id]]
            probabilities = torch.softmax(logits, dim=1)
            true_scores = probabilities[:, 0].detach().cpu().tolist()

        scores.extend(true_scores)

        finished = min(start + batch_size, len(text_pairs))
        print(f"scored {finished}/{len(text_pairs)} pairs")

    return scores


def make_cross_encoder_predictions(
    split_name,
    model_id=MODEL_ID,
    model_name=MODEL_NAME,
    batch_size=4,
):
    data = load_nevir_split(split_name)
    query_cases = make_query_cases(data)

    text_pairs = []
    for query_case in query_cases:
        text_pairs.append([query_case["query_text"], query_case["doc1_text"]])
        text_pairs.append([query_case["query_text"], query_case["doc2_text"]])

    scores = make_monot5_scores(text_pairs, model_id, batch_size)

    prediction_rows = []

    for i, query_case in enumerate(query_cases):
        doc1_score = float(scores[i * 2])
        doc2_score = float(scores[i * 2 + 1])

        prediction_row = make_prediction_row(
            split_name,
            model_name,
            query_case,
            doc1_score,
            doc2_score,
        )

        prediction_rows.append(prediction_row)

    return prediction_rows
