import faiss
import numpy as np

class VectorStore:
    def __init__(self, dimension: int):
        self.index = faiss.IndexFlatL2(dimension)
        self.chunks = []

    def add(self, embeddings, chunks):
        vectors = np.array(embeddings).astype("float32")
        self.index.add(vectors)
        self.chunks.extend(chunks)

    def search(self, embedding, k=5):
        query = np.array([embedding]).astype("float32")

        distances, indices = self.index.search(query, k)

        results = []

        for idx in indices[0]:
            if idx < len(self.chunks):
                results.append(self.chunks[idx])

        return results