import csv
import json
import math
import random
import re
import time
from collections import Counter
from pathlib import Path

import requests


PROJECT_DIR = Path(__file__).parent
DATA_PATH = PROJECT_DIR / "data" / "conll2003_ner.csv"
RESULTS_DIR = PROJECT_DIR / "results"
PREDICTIONS_PATH = RESULTS_DIR / "llm_pilot_predictions.csv"
METRICS_PATH = RESULTS_DIR / "llm_pilot_metrics.json"

MODEL_NAME = "llama3.2:3b"
RANDOM_SEED = 42
PILOT_SIZE = 500
TOP_K = 3

# Completed results, including failed requests, are saved here.
CHECKPOINT_PATH = RESULTS_DIR / f"llm_pilot_checkpoint_{PILOT_SIZE}.jsonl"

LABEL_MAP = {
    "PER": "PERSON",
    "ORG": "ORG",
    "LOC": "LOC",
    "MISC": "MISC",
}
VALID_LABELS = set(LABEL_MAP.values())


def tokenize(text):
    return re.findall(r"\w+", text.lower(), flags=re.UNICODE)


def make_terms(text):
    words = tokenize(text)
    bigrams = [
        f"{words[index]}_{words[index + 1]}"
        for index in range(len(words) - 1)
    ]
    return words + bigrams


def make_tfidf_vector(terms, idf):
    counts = Counter(terms)
    weights = {
        term: count * idf.get(term, 0.0)
        for term, count in counts.items()
    }
    norm = math.sqrt(sum(weight * weight for weight in weights.values()))

    if norm == 0:
        return {}

    return {term: weight / norm for term, weight in weights.items()}


def read_dataset():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    train_rows = []
    test_rows = []

    with DATA_PATH.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        required = {"split", "id", "sentence", "tokens", "ner_labels"}

        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(
                "CSV must include: split, id, sentence, tokens, ner_labels."
            )

        for row in reader:
            row["tokens"] = json.loads(row["tokens"])
            row["ner_labels"] = json.loads(row["ner_labels"])
            split = row["split"].strip().lower()

            if split == "train":
                train_rows.append(row)
            elif split == "test":
                test_rows.append(row)

    return train_rows, test_rows


def token_offsets(sentence, tokens):
    offsets = []
    cursor = 0

    for token in tokens:
        start = sentence.find(token, cursor)
        if start == -1:
            raise ValueError(f"Could not align token {token!r} in {sentence!r}")

        end = start + len(token)
        offsets.append((start, end))
        cursor = end

    return offsets


def bio_entities(sentence, tokens, tags):
    if len(tokens) != len(tags):
        raise ValueError("Token and BIO-label counts do not match.")

    offsets = token_offsets(sentence, tokens)
    entities = []
    active_start = None
    active_label = None

    def close_entity(last_token_index):
        nonlocal active_start, active_label

        if active_start is not None and last_token_index >= active_start:
            char_start = offsets[active_start][0]
            char_end = offsets[last_token_index][1]
            entities.append({
                "start": char_start,
                "end": char_end,
                "text": sentence[char_start:char_end],
                "label": LABEL_MAP[active_label],
            })

        active_start = None
        active_label = None

    for index, tag in enumerate(tags):
        if tag == "O" or "-" not in tag:
            close_entity(index - 1)
            continue

        prefix, raw_label = tag.split("-", maxsplit=1)

        if raw_label not in LABEL_MAP:
            close_entity(index - 1)
            continue

        if prefix == "B" or active_label != raw_label:
            close_entity(index - 1)
            active_start = index
            active_label = raw_label

    close_entity(len(tokens) - 1)
    return entities


def build_retrieval_index(train_rows):
    """Build retrieval documents using only the CoNLL training split."""
    examples = []

    for row in train_rows:
        entities = bio_entities(
            row["sentence"], row["tokens"], row["ner_labels"]
        )
        examples.append({
            "text": row["sentence"],
            "entities": [
                {"text": item["text"], "label": item["label"]}
                for item in entities
            ],
        })

    terms_per_example = [make_terms(example["text"]) for example in examples]
    document_frequency = Counter(
        term for terms in terms_per_example for term in set(terms)
    )

    document_count = len(examples)
    idf = {
        term: math.log((1 + document_count) / (1 + count)) + 1
        for term, count in document_frequency.items()
    }
    vectors = [
        make_tfidf_vector(terms, idf)
        for terms in terms_per_example
    ]

    return examples, idf, vectors


def retrieve_examples(text, examples, idf, vectors, top_k):
    query_vector = make_tfidf_vector(make_terms(text), idf)
    scored = []

    for index, vector in enumerate(vectors):
        similarity = sum(
            weight * vector.get(term, 0.0)
            for term, weight in query_vector.items()
        )
        scored.append((similarity, index))

    scored.sort(reverse=True)

    return [
        {**examples[index], "similarity": float(score)}
        for score, index in scored[:top_k]
    ]


def retrieve_context(retrieved):
    return json.dumps(
        [
            {"text": item["text"], "entities": item["entities"]}
            for item in retrieved
        ],
        ensure_ascii=False,
        indent=2,
    )


def ask_ollama(sentence, retrieved=None):
    """Try an Ollama request twice, limiting the generated response length."""
    retrieved = retrieved or []

    if retrieved:
        examples_text = retrieve_context(retrieved)
        instructions = (
            "Use the examples as guidance, but only label entities in the input."
        )
    else:
        examples_text = "(No examples provided.)"
        instructions = "Use your general language understanding."

    prompt = f"""
Perform named entity recognition.

{instructions}
Copy each entity's exact wording from the input.
Allowed labels: PERSON, ORG, LOC, MISC.
CoNLL-2003 has no DATE label, so do not return DATE.

Retrieved examples:
{examples_text}

Input:
{sentence}

Return only JSON in this format:
{{"entities": [{{"text": "entity phrase", "label": "ORG"}}]}}
If none are present, return {{"entities": []}}.
"""

    payload = {
        "model": MODEL_NAME,
        "system": (
            "You perform named entity recognition. "
            "Return only the requested JSON."
        ),
        "prompt": prompt,
        "format": "json",
        "stream": False,
        "options": {
            "temperature": 0,
            "num_predict": 128,
        },
    }

    start_time = time.perf_counter()
    last_error = None

    for attempt in range(1, 3):
        try:
            response = requests.post(
                "http://localhost:11434/api/generate",
                json=payload,
                timeout=(10, 60),
            )
            response.raise_for_status()

            result = json.loads(response.json()["response"])
            predictions = result.get("entities", [])

            if not isinstance(predictions, list):
                predictions = []

            return predictions, time.perf_counter() - start_time

        except (
            requests.exceptions.RequestException,
            KeyError,
            json.JSONDecodeError,
            TypeError,
        ) as error:
            last_error = error
            if attempt < 2:
                print(f"  Ollama request failed (attempt {attempt}/2): {error}")
                print("  Retrying once in 5 seconds...")
                time.sleep(5)

    raise RuntimeError(
        f"Ollama failed after 2 attempts. Last error: {last_error}"
    )


def safe_ask_ollama(sentence, retrieved=None):
    """Record a failed request as an empty prediction, then continue."""
    start_time = time.perf_counter()

    try:
        predictions, elapsed = ask_ollama(sentence, retrieved)
        return predictions, elapsed, ""
    except RuntimeError as error:
        return [], time.perf_counter() - start_time, str(error)


def prediction_spans(sentence, predictions):
    spans = set()
    used_occurrences = set()

    for prediction_index, item in enumerate(predictions):
        if not isinstance(item, dict):
            continue

        phrase = str(item.get("text", "")).strip()
        label = str(item.get("label", "")).upper()

        if not phrase:
            continue

        search_from = 0
        found = None

        while True:
            start = sentence.find(phrase, search_from)
            if start == -1:
                break

            end = start + len(phrase)
            if (start, end) not in used_occurrences:
                found = (start, end)
                break

            search_from = start + 1

        if found is None:
            spans.add((-prediction_index - 1, -prediction_index - 1, label))
        else:
            start, end = found
            used_occurrences.add((start, end))
            spans.add((start, end, label))

    return spans


def score_predictions(rows, prediction_field):
    totals = {"tp": 0, "fp": 0, "fn": 0}
    per_label = {
        label: {"tp": 0, "fp": 0, "fn": 0}
        for label in sorted(VALID_LABELS)
    }

    for row in rows:
        sentence = row["sentence"]
        gold = {
            (item["start"], item["end"], item["label"])
            for item in bio_entities(
                sentence, row["tokens"], row["ner_labels"]
            )
        }
        predicted = prediction_spans(sentence, row[prediction_field])

        for item in gold & predicted:
            totals["tp"] += 1
            per_label[item[2]]["tp"] += 1

        for item in predicted - gold:
            totals["fp"] += 1
            if item[2] in per_label:
                per_label[item[2]]["fp"] += 1

        for item in gold - predicted:
            totals["fn"] += 1
            per_label[item[2]]["fn"] += 1

    def finish(counts):
        tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        return {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            **counts,
        }

    return {
        "micro_exact_span": finish(totals),
        "per_label": {
            label: finish(counts)
            for label, counts in per_label.items()
        },
    }


def checkpoint_metadata(pilot_rows):
    return {
        "type": "metadata",
        "model": MODEL_NAME,
        "pilot_size": len(pilot_rows),
        "seed": RANDOM_SEED,
        "top_k": TOP_K,
        "sample_ids": [str(row["id"]) for row in pilot_rows],
    }


def load_checkpoint(expected_metadata):
    """Load completed predictions and repair any incomplete final line."""
    completed = {}

    if not CHECKPOINT_PATH.exists():
        with CHECKPOINT_PATH.open("w", encoding="utf-8") as file:
            file.write(json.dumps(expected_metadata) + "\n")
        return completed

    valid_records = []
    with CHECKPOINT_PATH.open("r", encoding="utf-8") as file:
        for line in file:
            try:
                valid_records.append(json.loads(line))
            except json.JSONDecodeError:
                # Ignore an incomplete line left by an interrupted write.
                continue

    if not valid_records or valid_records[0] != expected_metadata:
        raise RuntimeError(
            f"Checkpoint metadata does not match this run: {CHECKPOINT_PATH}\n"
            "Do not delete the checkpoint unless you intend to start over."
        )

    allowed_ids = set(expected_metadata["sample_ids"])
    for record in valid_records[1:]:
        if (
            record.get("type") == "result"
            and str(record.get("id")) in allowed_ids
        ):
            completed[str(record["id"])] = record

    # Rewrite only valid records so a broken final line cannot affect appends.
    with CHECKPOINT_PATH.open("w", encoding="utf-8") as file:
        file.write(json.dumps(expected_metadata) + "\n")
        for record in completed.values():
            file.write(json.dumps(record, ensure_ascii=False) + "\n")

    return completed


def append_checkpoint(record):
    with CHECKPOINT_PATH.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")
        file.flush()


def save_predictions(rows):
    fields = [
        "id",
        "sentence",
        "tokens",
        "ner_labels",
        "zero_shot_predictions",
        "rag_predictions",
        "zero_shot_seconds",
        "rag_seconds",
        "zero_shot_error",
        "rag_error",
    ]

    with PREDICTIONS_PATH.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()

        for row in rows:
            writer.writerow({
                "id": row["id"],
                "sentence": row["sentence"],
                "tokens": json.dumps(row["tokens"], ensure_ascii=False),
                "ner_labels": json.dumps(
                    row["ner_labels"], ensure_ascii=False
                ),
                "zero_shot_predictions": json.dumps(
                    row["zero_shot_predictions"], ensure_ascii=False
                ),
                "rag_predictions": json.dumps(
                    row["rag_predictions"], ensure_ascii=False
                ),
                "zero_shot_seconds": round(row["zero_shot_seconds"], 3),
                "rag_seconds": round(row["rag_seconds"], 3),
                "zero_shot_error": row.get("zero_shot_error", ""),
                "rag_error": row.get("rag_error", ""),
            })


def main():
    # Ensure this folder exists before loading or creating the checkpoint.
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    train_rows, test_rows = read_dataset()

    if not train_rows or not test_rows:
        raise ValueError("The CSV must contain train and test rows.")

    pilot_rows = random.Random(RANDOM_SEED).sample(
        test_rows,
        k=min(PILOT_SIZE, len(test_rows)),
    )

    metadata = checkpoint_metadata(pilot_rows)
    completed = load_checkpoint(metadata)
    examples, idf, vectors = build_retrieval_index(train_rows)

    print(f"Training retrieval examples: {len(examples)}")
    print(f"Test sentences in this run: {len(pilot_rows)}")
    print(f"Already checkpointed: {len(completed)}")
    print(f"Random seed: {RANDOM_SEED}")
    print(f"Model: {MODEL_NAME}")
    print("Running zero-shot and RAG predictions...")

    output_rows = []

    for number, row in enumerate(pilot_rows, start=1):
        sentence_id = str(row["id"])

        if sentence_id in completed:
            saved = completed[sentence_id]
            print(f"{number}/{len(pilot_rows)} resumed from checkpoint")
        else:
            sentence = row["sentence"]
            retrieved = retrieve_examples(
                sentence, examples, idf, vectors, TOP_K
            )

            # Each mode is handled separately. One failure won't stop the run.
            zero_shot, zero_time, zero_error = safe_ask_ollama(sentence)
            rag, rag_time, rag_error = safe_ask_ollama(sentence, retrieved)

            saved = {
                "type": "result",
                "id": sentence_id,
                "zero_shot_predictions": zero_shot,
                "rag_predictions": rag,
                "zero_shot_seconds": zero_time,
                "rag_seconds": rag_time,
                "zero_shot_error": zero_error,
                "rag_error": rag_error,
            }

            # Save this sentence even if an inference failed.
            append_checkpoint(saved)
            completed[sentence_id] = saved

            print(
                f"{number}/{len(pilot_rows)} complete "
                f"(zero-shot {zero_time:.1f}s, RAG {rag_time:.1f}s)"
            )

            if zero_error:
                print(f"  Zero-shot failure for sentence {sentence_id}: {zero_error}")
            if rag_error:
                print(f"  RAG failure for sentence {sentence_id}: {rag_error}")

        output_rows.append({
            **row,
            "zero_shot_predictions": saved["zero_shot_predictions"],
            "rag_predictions": saved["rag_predictions"],
            "zero_shot_seconds": saved["zero_shot_seconds"],
            "rag_seconds": saved["rag_seconds"],
            "zero_shot_error": saved.get("zero_shot_error", ""),
            "rag_error": saved.get("rag_error", ""),
        })

    metrics = {
        "model": MODEL_NAME,
        "pilot_size": len(output_rows),
        "seed": RANDOM_SEED,
        "top_k": TOP_K,
        "retrieval_source": "CoNLL-2003 train split only",
        "zero_shot": score_predictions(
            output_rows, "zero_shot_predictions"
        ),
        "rag": score_predictions(output_rows, "rag_predictions"),
        "zero_shot_failures": sum(
            bool(row["zero_shot_error"]) for row in output_rows
        ),
        "rag_failures": sum(
            bool(row["rag_error"]) for row in output_rows
        ),
        "mean_zero_shot_seconds": round(
            sum(row["zero_shot_seconds"] for row in output_rows)
            / len(output_rows),
            3,
        ),
        "mean_rag_seconds": round(
            sum(row["rag_seconds"] for row in output_rows)
            / len(output_rows),
            3,
        ),
        "note": (
            "Pilot run only. Failed inference requests are recorded as empty "
            "predictions and counted in the failure totals. Do not present "
            "these pilot results as final paper results."
        ),
    }

    save_predictions(output_rows)

    with METRICS_PATH.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    print("\nPilot exact-span results:")
    print(json.dumps(metrics, indent=2))
    print(f"\nSaved predictions: {PREDICTIONS_PATH}")
    print(f"Saved metrics: {METRICS_PATH}")
    print(f"Checkpoint retained at: {CHECKPOINT_PATH}")


if __name__ == "__main__":
    main()