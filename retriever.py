import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
from sentence_transformers import SentenceTransformer
import faiss


model = SentenceTransformer(
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)


def create_index(chunks):
    if not chunks:
        return None

    texts = [
        chunk["text"]
        for chunk in chunks
    ]

    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(
        dimension
    )

    index.add(
        embeddings.astype("float32")
    )

    return index


def search(
    query,
    chunks,
    index,
    top_k=5
):
    if not chunks or index is None:
        return []

    query_embedding = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    scores, indices = index.search(
        query_embedding.astype("float32"),
        min(top_k, len(chunks))
    )

    results = []

    for score, index_value in zip(
        scores[0],
        indices[0]
    ):

        results.append({
            "text": chunks[index_value]["text"],
            "page": chunks[index_value]["page"],
            "score": float(score)
        })

    return results