# Retrieval-Augmented Named Entity Recognition

A Streamlit project comparing three approaches to named entity recognition (NER):

1. A supervised `LinearSVC` model trained on CoNLL-2003.
2. Llama 3.2 prompted without retrieved examples (zero-shot).
3. Llama 3.2 prompted with similar labeled examples retrieved from the CoNLL-2003 training split (RAG).

The Llama model runs locally through Ollama. It is pretrained; this project does not train or fine-tune it.

## Dataset

The project uses the English CoNLL-2003 NER dataset. It has train, validation, and test splits, with entity labels for persons, organizations, locations, and miscellaneous entities. It has no DATE label.

The converted dataset is `data/conll2003_ner.csv`. The supervised model is trained using the training split. RAG retrieves examples from that split only. Evaluation uses a reproducible sample of 500 sentences from the test split, selected with random seed 42. The validation split is not used in the reported experiment.

The dataset's `PER` label is mapped to `PERSON` for evaluation and display.

Dataset source: [CoNLL-2003 on Hugging Face](https://huggingface.co/datasets/eriktks/conll2003).

## Methods

### Supervised baseline

`train_baseline.py` extracts word-level features, including capitalization, prefixes, suffixes, and neighboring words. A scikit-learn `DictVectorizer` converts the features into a sparse matrix, and `LinearSVC` predicts BIO labels for each token. The trained model is saved to `models/ner_baseline.joblib`.

### Llama zero-shot

`evaluate_llm.py` sends each test sentence to the locally running Llama 3.2 model without retrieved examples and asks it to return labeled entities in JSON.

### Llama with CoNLL RAG

The RAG method uses TF-IDF vectors and cosine similarity to retrieve the three most similar sentences from the CoNLL-2003 training split. It includes those labeled examples in the Llama prompt. Test sentences are never used as retrieval examples.

### Evaluation

Entities are evaluated using exact character-span matching: both the entity boundaries and label must match. The reported experiment uses the same 500 test sentences for all three methods.

## Pilot results

These are results from a 500-sentence pilot sample, not the full CoNLL-2003 test set.

| Method | Exact-span micro F1 | 95% bootstrap CI |
|---|---:|---:|
| Llama zero-shot | 0.4865 | 0.4444–0.5261 |
| Llama + CoNLL RAG | 0.5107 | 0.4694–0.5497 |
| LinearSVC baseline | 0.7104 | 0.6786–0.7436 |

The paired bootstrap estimate for RAG minus zero-shot F1 was `+0.0243`, with a 95% confidence interval of `−0.0114` to `+0.0627`. Because this interval includes zero, this pilot does not establish a reliable improvement from adding retrieval. The LinearSVC baseline scored higher on this sample.

There was one recorded inference failure in each Llama condition. Failed requests were retained as empty predictions and counted in the reported results. These results should be interpreted as pilot evidence, not as a definitive comparison.

Result files are in `results/`. `bootstrap_results.py` reproduces the paired sentence-bootstrap confidence intervals from the saved predictions.

## Run locally

Requirements: Python, Git, and [Ollama](https://ollama.com/download).

From the project folder, run these commands in PowerShell:

```powershell
py -m pip install -r requirements.txt
ollama pull llama3.2:3b
py train_baseline.py
py -m streamlit run app.py