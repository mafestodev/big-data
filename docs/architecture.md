# Implementation architecture

## Component relationships

```mermaid
flowchart LR
    subgraph Offline[Offline model development]
        D[WildlifeReID-10k] --> V[Validate and prepare metadata]
        V --> S[Similarity-aware split]
        S --> T[Swin + ArcFace training]
        T --> R[Selected checkpoint]
        R --> G[Gallery embedding builder]
        S --> G
        G --> I[Vector index]
    end

    subgraph Online[Online identification]
        U[User] --> W[Web interface]
        W --> A[Flask API]
        A --> P[Image preprocessing]
        P --> M[Swin embedding model]
        R --> M
        M --> Q[Cosine search]
        I --> Q
        Q --> A
        A --> W
    end
```

ArcFace is part of the training objective and is not used to select an identity
at runtime. The online model returns an L2-normalized embedding; the vector
index maps that representation to the closest known animal.

## Identification request

```mermaid
sequenceDiagram
    actor User
    participant Web as Browser interface
    participant API as Flask API
    participant Model as Swin model
    participant Index as Gallery index

    User->>Web: Upload wildlife image
    Web->>API: POST /api/v1/identify
    API->>API: Validate and preprocess
    API->>Model: Run normalized image tensor
    Model-->>API: 512-dimensional embedding
    API->>Index: Search by cosine similarity
    Index-->>API: Closest identity and score
    API-->>Web: JSON identification response
    Web-->>User: Show identity and similarity
```

## Artifact compatibility

The checkpoint, preprocessing configuration, embedding dimension, and gallery
index must be versioned together. When the model changes, regenerate the entire
gallery with the new checkpoint before deploying it.

## Delivery increments

1. Run the API with a fake identifier and confirm upload/error behavior.
2. Prepare a small, representative dataset sample and metadata format.
3. Complete the model training loop and validation metrics.
4. Build a gallery from the selected checkpoint.
5. Connect the real inference service to the Flask application.
6. Evaluate Top-1, Top-5, mAP, latency, and failure cases.

