# Retrieval-Augmented Named Entity Recognition

A Streamlit project that compares two named entity recognition (NER) approaches:

1. A supervised `LinearSVC` model trained on the CoNLL-2003 training split.
2. Retrieval-augmented few-shot prompting with Llama 3.2 through local Ollama.

## Dataset

The project uses the English CoNLL-2003 NER dataset. It contains train, validation, and test splits, with labels for persons, organizations, locations, and miscellaneous entities. The dataset does not have a DATE label.

The downloaded and converted CSV is `data/conll2003_ner.csv`. The training script uses the train split to fit the model and the test split to report metrics. It maps the dataset's `PER` label to `PERSON` in the app.

Dataset source: [CoNLL-2003 on Hugging Face](https://huggingface.co/datasets/eriktks/conll2003).

## Methods

### Supervised baseline

`train_baseline.py` extracts word and neighboring-token features, vectorizes them, and trains a scikit-learn `LinearSVC` token classifier. The trained model is saved at `models/ner_baseline.joblib`.

In the current run, the test results were **95.2% token accuracy** and **0.78 macro-average F1**. These are token-level metrics, not exact entity-span F1 scores. The large number of non-entity (`O`) tokens affects accuracy.

### Retrieval-augmented Llama

The app uses TF-IDF and cosine similarity to retrieve labeled examples from `data/examples.json`. It includes those examples in a prompt to the locally running Llama 3.2 model through Ollama. This mode is available when running the app on a computer with Ollama and the model installed.

The Llama model is pretrained; this project does not train or fine-tune it.

## Run locally

Requirements: Python, Git, and [Ollama for Windows](https://ollama.com/download).

From the project folder, run these commands in PowerShell:

```powershell
py -m pip install -r requirements.txt
ollama pull llama3.2:3b
py train_baseline.py
py -m streamlit run app.py
```