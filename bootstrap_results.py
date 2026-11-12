import csv
import json
import random
from pathlib import Path


PROJECT_DIR = Path(__file__).parent
RESULTS_DIR = PROJECT_DIR / "results"

LLM_PATH = RESULTS_DIR / "llm_pilot_predictions.csv"
BASELINE_PATH = RESULTS_DIR / "baseline_500_predictions.csv"
OUTPUT_PATH = RESULTS_DIR / "bootstrap_500_metrics.json"

BOOTSTRAP_REPEATS = 5000
SEED = 42

LABEL_MAP = {
    "PER": "PERSON",
    "ORG": "ORG",
    "LOC": "LOC",
    "MISC": "MISC",
}
VALID_LABELS = set(LABEL_MAP.values())


def parse_json(value):
    try:
        return json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return []


def token_offsets(sentence, tokens):
    """Use the same token-to-character alignment as evaluate_llm.py."""
    offsets = []
    cursor = 0

    for token in tokens:
        start = sentence.find(token, cursor)
        if start == -1:
            raise ValueError(
                f"Could not align token {token!r} in sentence {sentence!r}"
            )

        end = start + len(token)
        offsets.append((start, end))
        cursor = end

    return offsets


def bio_entities(sentence, tokens, tags):
    """Return gold or LinearSVC entity spans as character offsets."""
    if len(tokens) != len(tags):
        raise ValueError("Token and BIO-label counts do not match.")

    offsets = token_offsets(sentence, tokens)
    spans = set()
    active_start = None
    active_label = None

    def close_entity(last_token_index):
        nonlocal active_start, active_label

        if active_start is not None and last_token_index >= active_start:
            char_start = offsets[active_start][0]
            char_end = offsets[last_token_index][1]
            spans.add(
                (char_start, char_end, LABEL_MAP[active_label])
            )

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
    return spans


def prediction_spans(sentence, predictions):
    """Copy the evaluator's LLM phrase-to-character-span matching."""
    spans = set()
    used_occurrences = set()

    if isinstance(predictions, dict):
        predictions = predictions.get("entities", [])

    if not isinstance(predictions, list):
        predictions = []

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
            # Preserve the evaluator's treatment of an entity phrase
            # that does not appear in the input sentence.
            spans.add(
                (-prediction_index - 1, -prediction_index - 1, label)
            )
        else:
            start, end = found
            used_occurrences.add((start, end))
            spans.add((start, end, label))

    return spans


def load_data():
    if not LLM_PATH.exists():
        raise FileNotFoundError(f"Missing file: {LLM_PATH}")
    if not BASELINE_PATH.exists():
        raise FileNotFoundError(f"Missing file: {BASELINE_PATH}")

    with LLM_PATH.open("r", encoding="utf-8-sig", newline="") as file:
        llm_rows = {row["id"]: row for row in csv.DictReader(file)}

    with BASELINE_PATH.open("r", encoding="utf-8-sig", newline="") as file:
        baseline_rows = {
            row["id"]: row for row in csv.DictReader(file)
        }

    common_ids = sorted(set(llm_rows) & set(baseline_rows))
    if not common_ids:
        raise ValueError("No matching sentence IDs in the prediction files.")

    rows = []

    for sentence_id in common_ids:
        llm_row = llm_rows[sentence_id]
        baseline_row = baseline_rows[sentence_id]

        if llm_row["sentence"] != baseline_row["sentence"]:
            raise ValueError(f"Sentence mismatch for ID {sentence_id}")

        sentence = llm_row["sentence"]
        tokens = parse_json(llm_row["tokens"])
        gold_tags = parse_json(llm_row["ner_labels"])
        baseline_gold_tags = parse_json(
            baseline_row["gold_ner_labels"]
        )
        baseline_pred_tags = parse_json(
            baseline_row["predicted_ner_labels"]
        )

        gold = bio_entities(sentence, tokens, gold_tags)
        baseline_gold = bio_entities(
            sentence, tokens, baseline_gold_tags
        )

        if gold != baseline_gold:
            raise ValueError(
                f"Gold-label mismatch for sentence ID {sentence_id}"
            )

        rows.append({
            "id": sentence_id,
            "gold": gold,
            "zero_shot": prediction_spans(
                sentence,
                parse_json(llm_row["zero_shot_predictions"]),
            ),
            "rag": prediction_spans(
                sentence,
                parse_json(llm_row["rag_predictions"]),
            ),
            "svm": bio_entities(
                sentence, tokens, baseline_pred_tags
            ),
            "zero_shot_error": bool(
                llm_row.get("zero_shot_error", "").strip()
            ),
            "rag_error": bool(
                llm_row.get("rag_error", "").strip()
            ),
        })

    return rows


def counts_for_rows(rows, method):
    tp = fp = fn = 0

    for row in rows:
        gold = row["gold"]
        predicted = row[method]
        tp += len(gold & predicted)
        fp += len(predicted - gold)
        fn += len(gold - predicted)

    return {"tp": tp, "fp": fp, "fn": fn}


def f1_from_counts(counts):
    tp = counts["tp"]
    fp = counts["fp"]
    fn = counts["fn"]
    denominator = 2 * tp + fp + fn
    return 2 * tp / denominator if denominator else 0.0


def percentile(sorted_values, fraction):
    """Linearly interpolated percentile."""
    position = fraction * (len(sorted_values) - 1)
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = position - lower
    return (
        sorted_values[lower] * (1 - weight)
        + sorted_values[upper] * weight
    )


def confidence_interval(values):
    values = sorted(values)
    return [
        round(percentile(values, 0.025), 4),
        round(percentile(values, 0.975), 4),
    ]


def main():
    rows = load_data()
    rng = random.Random(SEED)

    methods = ["zero_shot", "rag", "svm"]
    names = {
        "zero_shot": "Llama zero-shot",
        "rag": "Llama + CoNLL RAG",
        "svm": "LinearSVC baseline",
    }

    point_scores = {
        method: f1_from_counts(counts_for_rows(rows, method))
        for method in methods
    }

    bootstrap_scores = {method: [] for method in methods}
    difference_scores = {
        "rag_minus_zero_shot": [],
        "svm_minus_rag": [],
    }

    for _ in range(BOOTSTRAP_REPEATS):
        sample = [
            rows[rng.randrange(len(rows))]
            for _ in range(len(rows))
        ]

        sample_f1 = {
            method: f1_from_counts(counts_for_rows(sample, method))
            for method in methods
        }

        for method in methods:
            bootstrap_scores[method].append(sample_f1[method])

        difference_scores["rag_minus_zero_shot"].append(
            sample_f1["rag"] - sample_f1["zero_shot"]
        )
        difference_scores["svm_minus_rag"].append(
            sample_f1["svm"] - sample_f1["rag"]
        )

    results = {
        "sentences": len(rows),
        "bootstrap_repetitions": BOOTSTRAP_REPEATS,
        "seed": SEED,
        "metric": "micro exact character-span F1",
        "methods": {},
        "paired_f1_differences": {},
        "zero_shot_failures": sum(
            row["zero_shot_error"] for row in rows
        ),
        "rag_failures": sum(row["rag_error"] for row in rows),
        "note": "500-sentence pilot estimate; not a full-test-set result.",
    }

    for method in methods:
        results["methods"][method] = {
            "name": names[method],
            "f1": round(point_scores[method], 4),
            "95_percentile_ci": confidence_interval(
                bootstrap_scores[method]
            ),
            **counts_for_rows(rows, method),
        }

    for comparison, values in difference_scores.items():
        results["paired_f1_differences"][comparison] = {
            "mean_difference": round(sum(values) / len(values), 4),
            "95_percentile_ci": confidence_interval(values),
        }

    print(f"Matched sentences: {len(rows)}")
    print(f"Bootstrap repetitions: {BOOTSTRAP_REPEATS}")
    print(f"Random seed: {SEED}")
    print("\nExact-span micro F1 and paired-bootstrap 95% CI:")

    for method in methods:
        result = results["methods"][method]
        low, high = result["95_percentile_ci"]
        print(
            f"{names[method]:20} "
            f"F1={result['f1']:.4f}  95% CI [{low:.4f}, {high:.4f}]"
        )

    print("\nPaired F1 differences and 95% CI:")
    for comparison, result in results["paired_f1_differences"].items():
        low, high = result["95_percentile_ci"]
        print(
            f"{comparison:24} "
            f"mean={result['mean_difference']:+.4f}  "
            f"95% CI [{low:+.4f}, {high:+.4f}]"
        )

    print(
        "\nRecorded inference errors: "
        f"zero-shot={results['zero_shot_failures']}, "
        f"RAG={results['rag_failures']}"
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)

    print(f"\nSaved matching bootstrap results: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()