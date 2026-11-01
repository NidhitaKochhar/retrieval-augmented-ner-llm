import html
import json
from pathlib import Path

import joblib
import requests
import spacy
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


MODEL_NAME = "llama3.2:3b"
PROJECT_DIR = Path(__file__).parent
EXAMPLES_PATH = PROJECT_DIR / "data" / "examples.json"
BASELINE_PATH = PROJECT_DIR / "models" / "ner_baseline.joblib"
VALID_LABELS = {"PERSON", "ORG", "LOC", "DATE", "MISC"}

st.set_page_config(
    page_title="Retrieval-Augmented NER",
    page_icon="🔎",
    layout="wide",
)


@st.cache_resource
def load_retrieval_index():
    """Load labeled examples and create a TF-IDF retrieval index."""
    with EXAMPLES_PATH.open("r", encoding="utf-8") as file:
        examples = json.load(file)

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        stop_words="english",
    )
    example_vectors = vectorizer.fit_transform(
        [example["text"] for example in examples]
    )
    return examples, vectorizer, example_vectors


@st.cache_resource
def load_trained_baseline():
    """Load the LinearSVC model trained on CoNLL-2003."""
    if not BASELINE_PATH.exists():
        raise FileNotFoundError(
            "Trained model not found. Run `py train_baseline.py` first."
        )
    return joblib.load(BASELINE_PATH)


@st.cache_resource
def load_english_tokenizer():
    """Create a tokenizer only; this does not load a pretrained NER model."""
    return spacy.blank("en")


def token_features(tokens, index):
    """Build the same word and context features used during training."""
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


def extract_entities_with_trained_model(text):
    """Predict BIO tags with the dataset-trained LinearSVC model."""
    saved = load_trained_baseline()
    model = saved["model"]
    vectorizer = saved["vectorizer"]

    doc = load_english_tokenizer()(text)
    tokens = [token.text for token in doc]

    if not tokens:
        return []

    features = [
        token_features(tokens, index)
        for index in range(len(tokens))
    ]
    X = vectorizer.transform(features)

    # Work around this environment's 64-bit sparse-index compatibility issue.
    X.indices = X.indices.astype("int32")
    X.indptr = X.indptr.astype("int32")

    predicted_tags = model.predict(X)

    label_map = {
        "PER": "PERSON",
        "ORG": "ORG",
        "LOC": "LOC",
        "MISC": "MISC",
    }

    entities = []
    active_start = None
    active_label = None

    def save_active_entity(last_token_index):
        nonlocal active_start, active_label

        if active_start is not None and last_token_index >= active_start:
            start_char = doc[active_start].idx
            end_char = doc[last_token_index].idx + len(doc[last_token_index])
            phrase = text[start_char:end_char]
            final_label = label_map[active_label]

            entities.append({"text": phrase, "label": final_label})

        active_start = None
        active_label = None

    for index, tag in enumerate(predicted_tags):
        if tag == "O":
            save_active_entity(index - 1)
            continue

        prefix, raw_label = tag.split("-", maxsplit=1)

        if raw_label not in label_map:
            save_active_entity(index - 1)
            continue

        if prefix == "B" or active_label != raw_label:
            save_active_entity(index - 1)
            active_start = index
            active_label = raw_label

    save_active_entity(len(tokens) - 1)
    return entities


def retrieve_examples(text, examples, vectorizer, example_vectors, top_k):
    """Return the labeled examples most similar to the input."""
    text_vector = vectorizer.transform([text])
    scores = cosine_similarity(text_vector, example_vectors)[0]
    best_indices = scores.argsort()[::-1][:top_k]

    return [
        {**examples[index], "similarity": float(scores[index])}
        for index in best_indices
    ]


def extract_entities_with_ollama(text, retrieved_examples):
    """Use retrieved labeled examples to guide local Ollama NER."""
    example_context = json.dumps(
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

    prompt = f"""
Extract named entities from the input text.

Use the retrieved examples as guidance, but only return entities that
appear in the input text. Copy each entity's exact wording from the input.
Use only these labels: PERSON, ORG, LOC, DATE, MISC.

Retrieved labeled examples:
{example_context}

Input text:
{text}

Return a JSON object in this format:
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

    parsed_output = json.loads(response.json()["response"])
    raw_entities = parsed_output.get("entities", [])

    if not isinstance(raw_entities, list):
        return []

    valid_entities = []
    seen = set()

    for entity in raw_entities:
        if not isinstance(entity, dict):
            continue

        phrase = str(entity.get("text", "")).strip()
        label = str(entity.get("label", "")).upper()

        if (
            phrase
            and phrase in text
            and label in VALID_LABELS
            and (phrase, label) not in seen
        ):
            valid_entities.append({"text": phrase, "label": label})
            seen.add((phrase, label))

    return valid_entities


st.title("Retrieval-Augmented Named Entity Recognition")
st.write(
    "Compare a supervised NER model trained on CoNLL-2003 with "
    "retrieval-augmented extraction using a local Llama model."
)

with st.sidebar:
    st.header("Settings")
    top_k = st.slider(
        "Similar examples to retrieve",
        min_value=1,
        max_value=5,
        value=3,
    )
    method = st.radio(
        "Extraction method",
        ["Dataset-trained baseline", "RAG + local Llama"],
    )
    st.markdown("**Labels:** PERSON · ORG · LOC · DATE · MISC")
    st.caption(
        "The CoNLL-2003 training labels cover PER, ORG, LOC, and MISC. "
        "The dataset-trained model does not learn DATE."
    )

sample_text = (
    "Microsoft CEO Satya Nadella announced a research center "
    "in London in 2025."
)
input_text = st.text_area(
    "Enter text to analyze",
    value=sample_text,
    height=150,
    max_chars=4000,
)

if st.button("Extract entities", type="primary"):
    text = input_text.strip()

    if not text:
        st.warning("Enter some text first.")
    else:
        try:
            examples, retrieval_vectorizer, example_vectors = (
                load_retrieval_index()
            )
            retrieved = retrieve_examples(
                text,
                examples,
                retrieval_vectorizer,
                example_vectors,
                top_k,
            )

            if method == "Dataset-trained baseline":
                with st.spinner("Predicting with the CoNLL-2003-trained model..."):
                    entities = extract_entities_with_trained_model(text)
                extraction_mode = "Supervised LinearSVC trained on CoNLL-2003"

            else:
                try:
                    with st.spinner("Analyzing with local Llama via Ollama..."):
                        entities = extract_entities_with_ollama(text, retrieved)
                    extraction_mode = (
                        "RAG few-shot prompting with local Llama 3.2 3B"
                    )
                except requests.RequestException:
                    with st.spinner(
                        "Ollama is unavailable; using the trained baseline..."
                    ):
                        entities = extract_entities_with_trained_model(text)
                    extraction_mode = (
                        "CoNLL-2003-trained LinearSVC fallback "
                        "(Ollama unavailable)"
                    )

            st.caption(f"Extraction mode: {extraction_mode}")
            st.subheader("Detected entities")

            if entities:
                st.dataframe(entities, hide_index=True, use_container_width=True)
            else:
                st.info("No entities were detected.")

            with st.expander("Retrieved examples", expanded=True):
                for number, example in enumerate(retrieved, start=1):
                    st.markdown(
                        f"**{number}. Similarity: "
                        f"{example['similarity']:.2f}**"
                    )
                    st.write(example["text"])
                    st.caption(
                        ", ".join(
                            f"{entity['text']} ({entity['label']})"
                            for entity in example["entities"]
                        )
                    )

        except Exception as error:
            st.error(f"Something went wrong: {error}")

st.divider()
st.caption(
    "CoNLL-2003 supervised baseline · TF-IDF retrieval · "
    "Few-shot prompting · Local Ollama inference · Streamlit"
)