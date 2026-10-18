#!/bin/sh

set -eu

PROJECT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
GUNICORN="$PROJECT_DIR/.venv/bin/gunicorn"
CHECKPOINT="$PROJECT_DIR/models/swin_embedding_model.pt"
GALLERY="$PROJECT_DIR/gallery/embeddings.npz"
DATASET="$PROJECT_DIR/data/archive"

if [ ! -x "$GUNICORN" ]; then
    echo "Gunicorn is not installed in .venv."
    echo "Run: source .venv/bin/activate && python -m pip install -e '.[dev]'"
    exit 1
fi

if [ ! -f "$CHECKPOINT" ]; then
    echo "Model checkpoint not found: $CHECKPOINT"
    exit 1
fi

if [ ! -f "$GALLERY" ]; then
    echo "Gallery embeddings not found: $GALLERY"
    exit 1
fi

if [ ! -d "$DATASET" ]; then
    echo "Wildlife dataset not found: $DATASET"
    echo "The dataset is required to display matched reference images."
    exit 1
fi

export WILDLIFE_DEVICE=${WILDLIFE_DEVICE:-cpu}
export WILDLIFE_CHECKPOINT_PATH=${WILDLIFE_CHECKPOINT_PATH:-$CHECKPOINT}
export WILDLIFE_GALLERY_PATH=${WILDLIFE_GALLERY_PATH:-$GALLERY}
export WILDLIFE_DATASET_PATH=${WILDLIFE_DATASET_PATH:-$DATASET}

cd "$PROJECT_DIR"

exec "$GUNICORN" \
    --bind 127.0.0.1:7860 \
    --workers 1 \
    --threads 1 \
    --timeout 300 \
    wildlife_reid.api.wsgi:app
