import faiss  # type: ignore
import numpy as np
from tqdm.auto import tqdm

from .document import Document


class FaissStore:
    def __init__(
        self, embedder, embd_size: int = 1536, max_text_size: int = 8192
    ) -> None:
        self.embedder = embedder
        index = faiss.IndexFlatIP(embd_size)
        index = faiss.IndexIDMap(index)
        self.index = index
        self.documents: list[Document] = []
        self.max_text_size = max_text_size

    def add(self, documents: list[Document], is_query: bool = False) -> None:
        smaller_documents = []
        for document in tqdm(documents):
            if len(document.text) > self.max_text_size:
                for i in range((len(document.text) // self.max_text_size) + 1):
                    text_chunk = document.text[
                        i * self.max_text_size : (i + 1) * self.max_text_size
                    ]
                    metadata = document.metadata.copy()
                    metadata["original-text"] = document.text
                    new_document = Document(text=text_chunk, metadata=metadata)
                    smaller_documents.append(new_document)
            else:
                smaller_documents.append(document)
        embeddings = self._embed(
            [document.text for document in smaller_documents], is_query=is_query
        )
        faiss.normalize_L2(embeddings)
        self.index.add_with_ids(
            embeddings,
            np.arange(
                len(self.documents),
                len(self.documents) + len(smaller_documents),
            ),
        )

        self.documents.extend(smaller_documents)

    def get(
        self, query: str, k: int = 5, is_query: bool = True
    ) -> tuple[list[str], list[float]]:
        references_embedding = self._embed([query], is_query=is_query)
        faiss.normalize_L2(references_embedding)
        distances, indices = self.index.search(references_embedding, k=k)
        return [self.documents[i] for i in indices[0]], distances[0]

    def _embed(self, texts: list[str], is_query: bool = False) -> np.ndarray:
        return self.embedder.embed(texts, is_query=is_query)
