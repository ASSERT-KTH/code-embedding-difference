This repository contains a pipeline for processing a modified version of the runbugrun dataset uploaded to HuggingFace, embedding buggy vs. fixed code pairs, and training a classifier to predict bug types.

Repository Structure
- load_dataset.ipynb – Loads the runbugrun dataset, tokenizes buggy and fixed functions in chunks for later processing.
- EMBEDDER2.ipynb – Embeds the tokenized buggy–fixed pairs, computes the difference vectors between them, and saves these difference embeddings as files.
- classifier.ipynb – Uses the difference vectors to perform multi-label classification with hyperparameter search, predicting which bug labels correspond to each buggy function.

Workflow
- Tokenization
Uses load_dataset.ipynb
Tokenizes buggy and fixed functions in chunks.
Prepares the data for embedding.
- Fixed - Buggy Difference Embedding
Uses EMBEDDER2.ipynb
Creates embeddings for each buggy and fixed function.
Computes the difference vector (fixed - buggy) for each pair.
Saves the difference vectors as files for later use.
- Bug Label Classification
Uses classifier.ipynb
Loads the saved difference vectors.
Runs a hyperparameter search to find the best model setup.
Trains a multi-label classifier to predict bug types (mapped to integers).

