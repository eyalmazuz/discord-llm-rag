import numpy as np
import tiktoken
import torch

from google import genai  # type: ignore
from google.genai import types
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm, trange  # type: ignore


class OpenAIEncoder:
    def __init__(
        self,
        model: str = "text-embedding-3-small",
        openai_key: str | None = None,
        embedding_size: int = 1536,
    ) -> None:
        self.client = OpenAI(api_key=openai_key)
        self.model = model
        self.embedding_size = embedding_size

    def embed(self, texts: list[str], **kwargs) -> np.ndarray:
        results: list[list[float]] = []
        total_tokens: int = 0
        for text in tqdm(texts):
            text = text.replace("\n", " ")
            total_tokens += self.num_tokens_from_string(text, "cl100k_base")
            response = self.client.embeddings.create(input=text, model=self.model)

            embedding = response.data[0].embedding
            if len(embedding) > self.embedding_size:
                embedding = self.normalize_l2(embedding[: self.embedding_size])
            results.append(embedding)

        print(f"Total tokens embedded: {total_tokens}")
        embeddings = np.array(results).astype(np.float32)
        # Check if the array is 1D and reshape if necessary
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)

        return embeddings

    def normalize_l2(self, x):
        x = np.array(x)
        if x.ndim == 1:
            norm = np.linalg.norm(x)
            if norm == 0:
                return x
            return x / norm
        else:
            norm = np.linalg.norm(x, 2, axis=1, keepdims=True)
            return np.where(norm == 0, x, x / norm).tolist()

    def num_tokens_from_string(self, string: str, encoding_name: str) -> int:
        """Returns the number of tokens in a text string."""
        encoding = tiktoken.get_encoding(encoding_name)
        num_tokens = len(encoding.encode(string))
        return num_tokens


class SentenceTransformerInstructEncoder:
    def __init__(
        self,
        model: str = "intfloat/multilingual-e5-large-instruct",
        embedidng_size: int = 1024,
        task_description: str | None = None,
    ) -> None:
        self.model = SentenceTransformer("intfloat/multilingual-e5-large-instruct", model_kwargs={"torch_dtype": "auto", "attn_implementation": "sdpa"})
        self.embedidng_size = embedidng_size
        if task_description:
            self.task_description = task_description
        else:
            self.task_description = "Given a question, retrieve relevant documents that best answer the question"

    def get_detailed_instruct(self, task_description: str, query: str) -> str:
        return f"Instruct: {task_description}\nQuery: {query}"

    # Each query must come with a one-sentence instruction that describes the task

    def embed(self, texts: list[str], is_query: bool = False, **kwargs) -> np.ndarray:
        for text in texts:
            text = text.replace("\n", " ")
        if is_query:
            text = self.get_detailed_instruct(self.task_description, text)
        embeddings = self.model.encode(
            texts,
            convert_to_tensor=False,
            normalize_embeddings=True,
            show_progress_bar=True,
        )

        embeddings = embeddings.astype(np.float32)
        # Check if the array is 1D and reshape if necessary
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)
        return embeddings


class GoogleEncoder:
    def __init__(
        self,
        model: str = "gemini-embedding-exp-03-07",
        api_key: str | None = None,
        embedidng_size: int = 1536,
    ) -> None:
        self.client = genai.Client(api_key=api_key)
        self.model = model
        self.embedidng_size = embedidng_size

    def embed(self, texts: list[str], **kwargs) -> np.ndarray:
        results: list[list[float]] = []
        for text in texts:
            text = text.replace("\n", " ")
        for i in trange(0, len(texts) // 100 + 1):
            response = self.client.models.embed_content(
                model=self.model,
                contents=texts[i * 100:(i + 1) * 100],
                config=types.EmbedContentConfig(task_type="QUESTION_ANSWERING")
            )

            embedding = [response.embeddings[i].values for i in range(len(texts[i * 100:(i + 1) * 100]))]
            results.extend(embedding)

        embeddings = np.array(results).astype(np.float32)
        # Check if the array is 1D and reshape if necessary
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)

        return embeddings
