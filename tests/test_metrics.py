import numpy as np
from wildlife_reid.training.metrics import evaluate


# PLACEHOLDER DATA
# These are fake embeddings ONLY to test the evaluation
# framework before real model is available.


query_embeddings = np.array([
    [1.0, 0.0, 0.0],   
    [0.0, 1.0, 0.0],   
    [0.0, 0.0, 1.0],   
    [0.9, 0.1, 0.0],   
], dtype=np.float32)

query_ids = np.array([
    "A",
    "B",
    "C",
    "A",
])

gallery_embeddings = np.array([
    [1.0, 0.0, 0.0],   # A
    [0.0, 1.0, 0.0],   # B
    [0.0, 0.0, 1.0],   # C
    [0.8, 0.2, 0.0],   # A
    [0.0, 0.8, 0.2],   # B
    [0.1, 0.0, 0.9],   # C
], dtype=np.float32)

gallery_ids = np.array([
    "A",
    "B",
    "C",
    "A",
    "B",
    "C",
])


results = evaluate(
    query_embeddings=query_embeddings,
    query_ids=query_ids,
    gallery_embeddings=gallery_embeddings,
    gallery_ids=gallery_ids,
)

print("=== WildlifeReID Evaluation Test ===")
print(f"Top-1 Accuracy: {results['top_1_accuracy']:.4f}")
print(f"Top-5 Accuracy: {results['top_5_accuracy']:.4f}")
print(f"mAP:            {results['mAP']:.4f}")

print("\nRankings:")
for i, (true_id, ranking) in enumerate(zip(query_ids, results["rankings"]), start=1):
    print(f"Query {i} | True ID: {true_id} | Ranking: {ranking}")
