import joblib
from pathlib import Path
import csv
import json
from collections import Counter

from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.svm import LinearSVC


DATA_PATH = "data/conll2003_ner.csv"


def load_dataset():
    dataset = {}

    with open(DATA_PATH, "r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)

        for row in reader:
            split = row["split"].strip().lower()
            tokens = json.loads(row["tokens"])
            labels = json.loads(row["ner_labels"])

            dataset.setdefault(split, []).append((tokens, labels))

    return dataset


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


def make_training_data(sentences):
    features = []
    labels = []

    for tokens, tags in sentences:
        for index, tag in enumerate(tags):
            features.append(token_features(tokens, index))
            labels.append(tag)

    return features, labels


data = load_dataset()

print("Dataset splits:", {name: len(rows) for name, rows in data.items()})
print("Training label counts:", Counter(
    tag
    for _, labels in data["train"]
    for tag in labels
))

train_features, train_labels = make_training_data(data["train"])
test_features, test_labels = make_training_data(data["test"])

vectorizer = DictVectorizer(sparse=True)
X_train = vectorizer.fit_transform(train_features)
X_test = vectorizer.transform(test_features)

# LinearSVC requires 32-bit sparse-matrix indices
for matrix in (X_train, X_test):
    matrix.indices = matrix.indices.astype("int32")
    matrix.indptr = matrix.indptr.astype("int32")

model = LinearSVC(class_weight="balanced")
model.fit(X_train, train_labels)

predictions = model.predict(X_test)

print("\nTest token accuracy:", round(accuracy_score(test_labels, predictions), 4))
print("\nTest classification report:")
print(classification_report(test_labels, predictions, zero_division=0))

model_dir = Path("models")
model_dir.mkdir(exist_ok=True)

joblib.dump(
    {"model": model, "vectorizer": vectorizer},
    model_dir / "ner_baseline.joblib"
)

print("\nSaved trained model to models/ner_baseline.joblib")