import csv
import json
import math
import re
from collections import Counter
from pathlib import Path

import requests
import streamlit as st


MODEL_NAME = "llama3.2:3b"
PROJECT_DIR = Path(__file__).parent
DATA_PATH = PROJECT_DIR / "data" / "conll2003_ner.csv"

# CoNLL-2003 labels. PER is displayed as PERSON.
LABEL_MAP = {
    "PER": "PERSON",
    "ORG": "ORG",
    "LOC": "LOC",
    "MISC": "MISC",
}
VALID_LABELS = set(LABEL_MAP.values())


st.set_page_config(
    page_title="CoNLL Retrieval-Augmented NER",
    page_icon="🔎",
    layout="wide",
)


def tokenize(text):
    """Tokenize words for plain-Python TF-IDF retrieval."""
    return re.findall(r"\w+", text.lower(), flags=re.UNICODE)


def make_terms(text):
    """Use word unigrams and adjacent word bigrams."""
    words = tokenize(text)
    bigrams = [
        f"{words[index]}_{words[index + 1]}"
        for index in range(len(words) - 1)
    ]
    return words + bigrams


def entities_from_bio(tokens, tags):
    """Convert BIO token labels into labeled entity phrases."""
    entities = []
    start = None
    active_label = None

    def close_entity(end_index):
        nonlocal start, active_label

        if start is not None and end_index > start:
            phrase = " ".join(tokens[start:end_index])
            entities.append({
                "text": phrase,
                "label": LABEL_MAP[active_label],
            })

        start = None
        active_label = None

    for index, tag in enumerate(tags):
        if tag == "O" or "-" not in tag:
            close_entity(index)
            continue

        prefix, raw_label = tag.split("-", maxsplit=1)

        if raw_label not in LABEL_MAP:
            close_entity(index)
            continue

        if prefix == "B" or active_label != raw_label:
            close_entity(index)
            start = index
            active_label = raw_label

    close_entity(len(tokens))
    return entities


def make_tfidf_vector(terms, inverse_document_frequency):
    """Create a normalized TF-IDF vector using only the Python standard library."""
    counts = Counter(terms)
    weights = {
        term: count * inverse_document_frequency.get(term, 0.0)
        for term, count in counts.items()
    }

    norm = math.sqrt(sum(weight * weight for weight in weights.values()))
    if norm == 0:
        return {}

    return {
        term: weight / norm
        for term, weight in weights.items()
    }


@st.cache_resource
def load_training_retrieval_index():
    """Load CoNLL training sentences only and build the retrieval index."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset CSV not found: {DATA_PATH}")

    examples = []

    with DATA_PATH.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)

        required_columns = {"split", "tokens", "ner_labels"}
        if not reader.fieldnames or not required_columns.issubset(reader.fieldnames):
            raise ValueError(
                "CSV must contain these columns: split, tokens, ner_labels."
            )

        for row in reader:
            if row["split"].strip().lower() != "train":
                continue

            tokens = json.loads(row["tokens"])
            tags = json.loads(row["ner_labels"])

            if len(tokens) != len(tags):
                continue

            sentence = row.get("sentence", "").strip()
            if not sentence:
                sentence = " ".join(tokens)

            entities = entities_from_bio(tokens, tags)

            examples.append({
                "text": sentence,
                "entities": entities,
            })

    if not examples:
        raise ValueError("No training examples were found in the dataset CSV.")

    tokenized = [make_terms(example["text"]) for example in examples]

    document_frequency = Counter(
        term
        for terms in tokenized
        for term in set(terms)
    )

    document_count = len(examples)
    idf = {
        term: math.log((1 + document_count) / (1 + frequency)) + 1
        for term, frequency in document_frequency.items()
    }

    vectors = [
        make_tfidf_vector(terms, idf)
        for terms in tokenized
    ]

    return examples, idf, vectors


def retrieve_examples(text, examples, idf, vectors, top_k):
    """Retrieve the training examples with greatest TF-IDF cosine similarity."""
    query_vector = make_tfidf_vector(make_terms(text), idf)
    scored = []

    for index, example_vector in enumerate(vectors):
        similarity = sum(
            weight * example_vector.get(term, 0.0)
            for term, weight in query_vector.items()
        )
        scored.append((similarity, index))

    scored.sort(reverse=True)

    return [
        {
            **examples[index],
            "similarity": float(score),
        }
        for score, index in scored[:top_k]
    ]


def extract_entities_with_ollama(text, retrieved_examples):
    """Run zero-shot Llama or retrieval-augmented Llama through Ollama."""
    if retrieved_examples:
        context = json.dumps(
            [
                {
                    "text": example["text"],
                    "entities": example["entities"],
                }
                for example in retrieved_examples
            ],
            ensure_ascii=False,
            indent=2,
        )
        example_instruction = (
            "Use the retrieved labeled examples as guidance. "
            "Do not copy entities from them unless they occur in the input."
        )
    else:
        context = "(No examples supplied.)"
        example_instruction = (
            "Extract entities using your general language understanding. "
            "No labeled examples are supplied."
        )

    prompt = f"""
Identify named entities in the input.

{example_instruction}
Copy entity text exactly from the input.
Use only these labels: PERSON, ORG, LOC, MISC.
Do not return DATE because CoNLL-2003 does not contain a DATE label.

Examples:
{context}

Input:
{text}

Return only JSON in this format:
{{"entities": [{{"text": "entity phrase", "label": "ORG"}}]}}
If there are no entities, return {{"entities": []}}.
"""

    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": MODEL_NAME,
            "system": (
                "You perform named entity recognition. "
                "Return only the requested JSON object."
            ),
            "prompt": prompt,
            "format": "json",
            "stream": False,
            "options": {"temperature": 0},
        },
        timeout=180,
    )
    response.raise_for_status()

    result = json.loads(response.json()["response"])
    raw_entities = result.get("entities", [])

    if not isinstance(raw_entities, list):
        return []

    entities = []
    seen = set()

    for item in raw_entities:
        if not isinstance(item, dict):
            continue

        phrase = str(item.get("text", "")).strip()
        label = str(item.get("label", "")).upper()

        if (
            phrase
            and phrase in text
            and label in VALID_LABELS
            and (phrase, label) not in seen
        ):
            entities.append({"text": phrase, "label": label})
            seen.add((phrase, label))

    return entities


st.title("Retrieval-Augmented Named Entity Recognition")
st.write(
    "Compare zero-shot Llama with retrieval-augmented Llama. "
    "The RAG method retrieves examples only from the CoNLL-2003 training split."
)

with st.sidebar:
    st.header("Experiment settings")

    method = st.radio(
        "Method",
        ["Llama zero-shot", "Llama + CoNLL RAG"],
    )

    top_k = st.slider(
        "Training examples to retrieve",
        min_value=1,
        max_value=5,
        value=3,
        disabled=(method == "Llama zero-shot"),
    )

    st.markdown("**CoNLL labels:** PERSON · ORG · LOC · MISC")
    st.caption(
        "This dataset has no DATE label. Test examples are not used for retrieval."
    )

input_text = st.text_area(
    "Enter text to analyze",
    value="Microsoft CEO Satya Nadella announced a research center in London.",
    height=150,
    max_chars=4000,
)

if st.button("Extract entities", type="primary"):
    text = input_text.strip()

    if not text:
        st.warning("Enter some text first.")
    else:
        try:
            examples, idf, vectors = load_training_retrieval_index()

            if method == "Llama + CoNLL RAG":
                retrieved = retrieve_examples(
                    text,
                    examples,
                    idf,
                    vectors,
                    top_k,
                )
            else:
                retrieved = []

            with st.spinner(f"Analyzing with {MODEL_NAME} through Ollama..."):
                entities = extract_entities_with_ollama(text, retrieved)

            st.caption(f"Method: {method} · Retrieved corpus: CoNLL train split")
            st.subheader("Detected entities")

            if entities:
                st.dataframe(
                    entities,
                    hide_index=True,
                    width="stretch",
                )
            else:
                st.info("No entities were detected.")

            if retrieved:
                with st.expander("Retrieved CoNLL training examples"):
                    for number, example in enumerate(retrieved, start=1):
                        st.markdown(
                            f"**{number}. Similarity: "
                            f"{example['similarity']:.3f}**"
                        )
                        st.write(example["text"])
                        st.caption(
                            ", ".join(
                                f"{entity['text']} ({entity['label']})"
                                for entity in example["entities"]
                            )
                            or "No entities"
                        )

        except requests.RequestException as error:
            st.error(
                "Could not connect to Ollama. Make sure Ollama is running "
                f"and {MODEL_NAME} is installed. Details: {error}"
            )
        except Exception as error:
            st.error(f"Something went wrong: {error}")


st.divider()
st.caption(
    "CoNLL-2003 training-split retrieval · TF-IDF · "
    "Zero-shot vs RAG · Local Ollama inference"
)