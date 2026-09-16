# Wildlife Re-Identification

An end-to-end prototype for identifying individual animals from images using a
pretrained Swin Transformer, ArcFace metric learning, and cosine-similarity
retrieval.

## Architecture

The project contains two separate workflows:

1. **Offline:** validate and split WildlifeReID-10k, fine-tune Swin with
   ArcFace, select a checkpoint, and build the gallery embeddings.
2. **Online:** accept an uploaded image, create its normalized embedding,
   search the gallery, and return the closest animal identity.

During training, ArcFace shapes the embedding space. During inference, the
ArcFace classification head is not used; prediction is performed by cosine
similarity against the gallery.

## Project layout

```text
src/wildlife_reid/
  api/          Flask API and browser interface
  gallery/      Gallery creation and vector search
  training/     Dataset, model, ArcFace, training and evaluation
  common/       Configuration and shared utilities
data/           Dataset link, samples, and generated split metadata
models/         Trained checkpoints and model metadata
gallery/        Generated gallery embeddings and identity metadata
tests/          Unit and API tests
```

## Eli - How to test it

Install NumPy if necessary:
In terminal type "pip install numpy"

Then run "python tests/test_metrics.py"

The test uses fake embeddings and fake individual IDs to confirm that the evaluation calculations work.

## Quick start

Create a virtual environment and install the project:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
```

Run the placeholder API:

```bash
wildlife-api
```

Then open `http://127.0.0.1:5000`. The health endpoint works immediately.
Identification becomes available after a compatible model checkpoint and
gallery have been generated.

Run tests:

```bash
pytest
```

## Dataset

Download WildlifeReID-10k using the URL in `data/dataset_url.txt`. Do not
commit the full dataset or trained weights to Git.

## Initial technical choices

- Backbone: `swin_tiny_patch4_window7_224`
- Input resolution: 224 x 224 RGB
- Embedding dimension: 512
- Training loss: ArcFace
- Retrieval: exact cosine similarity (replaceable with FAISS)
- Serving: Flask REST API
- Initial scope: closed-set individual identification

