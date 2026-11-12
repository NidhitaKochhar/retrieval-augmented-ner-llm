import csv
import json
from pathlib import Path

import joblib


PROJECT_DIR = Path(__file__).parent
DATA_PATH = PROJECT_DIR / "data" / "conll2003_ner.csv"
MODEL_PATH = PROJECT_DIR / "models" / "ner_baseline.joblib"
LLM_PREDICTIONS_PATH = PROJECT_DIR / "results" / "llm_pilot_predictions.csv"
METRICS_PATH = PROJECT_DIR / "results" / "baseline_500_metrics.json"
PREDICTIONS_PATH = PROJECT_DIR / "results" / "baseline_500_predictions.csv"

LABELS = ["PERSON", "ORG", "LOC", "MISC"]


def token_features(tokens, index):
    word = tokens[index]

    features = {
        "word.lower": word.lower(),
        "word.isupper": word.isupper(),
        "word.istitle": word.istitle(),
        "word.isdigit": word.isdigit(),
        "word.prefix2": word[:2].lower(),
        "word.prefix3": word[:3].lower(),
        "word.suffix2": word[-2:].lower(),
        "word.suffix3": word[-3:].lower(),
    }

    if index == 0:
        features["previous"] = "<START>"
    else:
        previous = tokens[index - 1]
        features["previous.lower"] = previous.lower()
        features["previous.istitle"] = previous.istitle()

    if index == len(tokens) - 1:
        features["next"] = "<END>"
    else:
        following = tokens[index + 1]
        features["next.lower"] = following.lower()
        features["next.istitle"] = following.istitle()

    return features


def spans_from_bio(tags, sentence_id):
    """Convert BIO tags to exact token-span tuples."""
    spans = set()
    active_label = None
    start = None

    def close_span(end):
        if active_label is not None:
            label = "PERSON" if active_label == "PER" else active_label
            spans.add((str(sentence_id), start, end, label))

    for index, tag in enumerate(tags):
        if tag == "O" or "-" not in tag:
            close_span(index)
            active_label = None
            start = None
            continue

        prefix, label = tag.split("-", 1)

        if prefix == "B" or active_label != label:
            close_span(index)
            active_label = label
            start = index

    close_span(len(tags))
    return spans


def calculate_metrics(gold_spans, predicted_spans):
    true_positive = len(gold_spans & predicted_spans)
    false_positive = len(predicted_spans - gold_spans)
    false_negative = len(gold_spans - predicted_spans)

    precision = (
        true_positive / (true_positive + false_positive)
        if true_positive + false_positive else 0.0
    )
    recall = (
        true_positive / (true_positive + false_negative)
        if true_positive + false_negative else 0.0
    )
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0.0
    )

    per_label = {}

    for label in LABELS:
        gold = {span for span in gold_spans if span[3] == label}
        predicted = {
            span for span in predicted_spans if span[3] == label
        }

        tp = len(gold & predicted)
        fp = len(predicted - gold)
        fn = len(gold - predicted)

        label_precision = tp / (tp + fp) if tp + fp else 0.0
        label_recall = tp / (tp + fn) if tp + fn else 0.0
        label_f1 = (
            2 * label_precision * label_recall
            / (label_precision + label_recall)
            if label_precision + label_recall else 0.0
        )

        per_label[label] = {
            "precision": round(label_precision, 4),
            "recall": round(label_recall, 4),
            "f1": round(label_f1, 4),
            "tp": tp,
            "fp": fp,
            "fn": fn,
        }

    return {
        "micro_exact_span": {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "tp": true_positive,
            "fp": false_positive,
            "fn": false_negative,
        },
        "per_label": per_label,
    }


def main():
    required_paths = (
        DATA_PATH,
        MODEL_PATH,
        LLM_PREDICTIONS_PATH,
    )
    for path in required_paths:
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path}")

    # Use the exact sentence IDs from the completed Llama evaluation.
    with LLM_PREDICTIONS_PATH.open(
        "r", encoding="utf-8-sig", newline=""
    ) as file:
        reference_rows = list(csv.DictReader(file))

    if not reference_rows:
        raise ValueError(
            f"No prediction rows found in {LLM_PREDICTIONS_PATH}"
        )

    # Index the CoNLL test split by ID and sentence.
    test_rows = {}

    with DATA_PATH.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)

        for row in reader:
            if row["split"].strip().lower() == "test":
                key = (str(row["id"]), row["sentence"])
                test_rows[key] = row

    sample = []

    for reference in reference_rows:
        key = (str(reference["id"]), reference["sentence"])

        if key not in test_rows:
            raise ValueError(
                "Could not match a Llama prediction row to the CoNLL "
                "test data: "
                f"id={reference['id']}, "
                f"sentence={reference['sentence']!r}"
            )

        sample.append(test_rows[key])

    if len(sample) != len(reference_rows):
        raise ValueError("The matched baseline sample size is incorrect.")

    saved_model = joblib.load(MODEL_PATH)
    model = saved_model["model"]
    vectorizer = saved_model["vectorizer"]

    gold_spans = set()
    predicted_spans = set()
    prediction_rows = []

    for row in sample:
        tokens = json.loads(row["tokens"])
        gold_tags = json.loads(row["ner_labels"])

        features = [
            token_features(tokens, index)
            for index in range(len(tokens))
        ]
        X = vectorizer.transform(features)
        predicted_tags = model.predict(X)

        sentence_id = str(row["id"])
        predicted_tags = [str(tag) for tag in predicted_tags]

        prediction_rows.append({
            "id": sentence_id,
            "sentence": row["sentence"],
            "tokens": json.dumps(tokens, ensure_ascii=False),
            "gold_ner_labels": json.dumps(
                gold_tags, ensure_ascii=False
            ),
            "predicted_ner_labels": json.dumps(
                predicted_tags, ensure_ascii=False
            ),
        })

        gold_spans.update(spans_from_bio(gold_tags, sentence_id))
        predicted_spans.update(
            spans_from_bio(predicted_tags, sentence_id)
        )

    results = calculate_metrics(gold_spans, predicted_spans)
    results["model"] = "LinearSVC trained on CoNLL-2003 train split"
    results["sentence_count"] = len(sample)
    results["evaluation_ids_source"] = (
        "results/llm_pilot_predictions.csv"
    )
    results["metric"] = "Exact entity-span match"

    print("\nLinearSVC exact-span results on the Llama evaluation sentences")
    print(f"Sentences: {len(sample)}")

    micro = results["micro_exact_span"]
    print(f"Micro precision: {micro['precision']:.4f}")
    print(f"Micro recall:    {micro['recall']:.4f}")
    print(f"Micro F1:        {micro['f1']:.4f}")
    print(f"TP={micro['tp']}, FP={micro['fp']}, FN={micro['fn']}")

    print("\nPer-label exact-span results:")
    for label in LABELS:
        values = results["per_label"][label]
        print(
            f"{label:7} "
            f"precision={values['precision']:.4f} "
            f"recall={values['recall']:.4f} "
            f"F1={values['f1']:.4f}"
        )

    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)

    with METRICS_PATH.open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)

    with PREDICTIONS_PATH.open(
        "w", encoding="utf-8", newline=""
    ) as file:
        fields = [
            "id",
            "sentence",
            "tokens",
            "gold_ner_labels",
            "predicted_ner_labels",
        ]
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(prediction_rows)

    print(f"\nSaved baseline metrics: {METRICS_PATH}")
    print(f"Saved sentence predictions: {PREDICTIONS_PATH}")
    print(
        "\nThis is a 500-sentence pilot comparison, "
        "not a final paper result."
    )


if __name__ == "__main__":
    main()