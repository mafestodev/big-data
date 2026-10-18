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

## Quick start

Create a virtual environment and install the project:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
```

Run the application:

```bash
wildlife-api
```

Then open `http://127.0.0.1:5000`. Identification is available when compatible
model and gallery artifacts exist at the configured paths.

Run tests:

```bash
pytest
```

## Dataset

Download WildlifeReID-10k using the URL in `data/dataset_url.txt`. Do not
commit the full dataset or trained weights to Git.

## Demo deployment

The demonstration is published from the local machine through a Cloudflare
Quick Tunnel. Cloudflare supplies a temporary public HTTPS address without
requiring a Cloudflare account, domain, or cloud-compute subscription.

The model, gallery embeddings, and dataset remain on the demonstration
computer. The API uses the gallery's relative image path to serve the matched
reference image through the tunnel together with the identity, species, and
similarity result.

Required local artifacts:

```text
models/swin_embedding_model.pt
gallery/embeddings.npz
data/archive/images/...
```

Install the deployment dependencies once:

```bash
source .venv/bin/activate
python -m pip install -e '.[dev]'
brew install cloudflared
```

Start the production application in one terminal:

```bash
./scripts/run_demo_server.sh
```

After `/health` reports that the model is ready, start the public tunnel in a
second terminal:

```bash
./scripts/start_quick_tunnel.sh
```

Open the printed `https://*.trycloudflare.com` URL from another device. The URL
remains active while both commands and the host computer continue running.

To restrict the tunnel to one email address:

```bash
TUNNEL_ALLOWED_EMAIL="name@example.com" ./scripts/start_quick_tunnel.sh
```

See `docs/deployment.md` for validation and presentation guidance.

## Container validation

The Docker configuration remains available as evidence that the application is
container-ready:

```bash
docker build -t wildlife-reid .
docker run --rm -p 7860:7860 wildlife-reid
```

## Initial technical choices

- Backbone: `swin_tiny_patch4_window7_224`
- Input resolution: 224 x 224 RGB
- Embedding dimension: 512
- Training loss: ArcFace
- Retrieval: exact cosine similarity 
- Serving: Flask REST API
- Initial scope: closed-set individual identification

