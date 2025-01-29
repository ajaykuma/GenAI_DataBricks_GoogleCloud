from docarray import Document, DocumentArray
from jina import Executor, requests, Flow

from sentence_transformers import SentenceTransformer
import torch


class SentenceEncoder(Executor):
    """A simple sentence encoder that can be run on a CPU or a GPU

    :param device: The pytorch device that the model is on, e.g. 'cpu', 'cuda', 'cuda:1'
    """

    def __init__(self, device: str = 'cpu', *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
        self.model.to(device)  # Move the model to device

    @requests
    def encode(self, docs: DocumentArray, **kwargs):
        """Add text-based embeddings to all documents"""
        with torch.inference_mode():
            embeddings = self.model.encode(docs.texts, batch_size=32)
        docs.embeddings = embeddings

