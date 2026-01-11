This repository contains a pipeline for processing a modified version of the RunBugRun dataset on HuggingFace, embedding buggy and fixed code pairs, computing difference embeddings, training models to map buggy to fixed embeddings, and performing multi-label classification to predict bug types.


## Datasets on Huggingface

https://huggingface.co/datasets/ASSERT-KTH/RunBugRun-Final

## Trained models on Huggingface

https://huggingface.co/ASSERT-KTH/Buggy-to-Fixed-Code-ViT1D

## Repository Structure
- tokenize_dataset.ipynb – Tokenizes the RunBugRun dataset that can be found on HuggingFace and saves it locally.
- embedder_buggy_fixed.ipynb - Embeds the tokenized buggy-fixed pairs and stores both embeddings.
- embedder_diff.ipynb – Embeds the tokenized buggy–fixed pairs, computes the difference vectors between them, and saves these difference embeddings as files.
- cpp_bug_classifier.ipynb – Uses the difference vectors to perform multi-label classification with hyperparameter search, predicting which bug labels correspond to each buggy function.
- MLP_projection.ipynb - Trains an MLP to map buggy to fixed vectors in embedding space.
- ViT_projectin.ipynb - Trains a 1D-ViT to map buggy to fixed vectors in embedding space. Includes data analysis on model perfromance progression, cosine similarity between fixed and predicted vectors, as well as performance for predicting bug labels using predicted-buggy vs fixed-buggy.

## Workflow

### 1. Tokenization
- Uses `tokenize_dataset.ipynb`.
- Tokenizes buggy and fixed functions in chunks.
- Prepares the data for embedding.

### 2. Embedding and Difference Vectors
- Uses `embedder_buggy_fixed.ipynb` to create embeddings for each buggy and fixed function.
- Optionally, `embedder_diff.ipynb` computes difference vectors (`fixed - buggy`) for each pair.
- Saves embeddings and/or difference vectors as files for later use.

### 3. Bug Label Classification
- Uses `cpp_bug_classifier.ipynb`.
- Loads the saved difference vectors.
- Performs hyperparameter search to find the best model configuration.
- Trains a multi-label classifier to predict bug types (mapped to integers).

### 4. Buggy → Fixed Embedding Mapping
- Uses `MLP_projection.ipynb` or `ViT_projection.ipynb`.
- Trains a model to map buggy embeddings to fixed embeddings.
- `ViT_projection.ipynb` includes:
  - Performance analysis over training progression
  - Cosine similarity between predicted and actual fixed embeddings
  - Evaluation of predicted embeddings for bug label classification
