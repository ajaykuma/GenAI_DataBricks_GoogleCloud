import torch

from docarray import DocumentArray,Document
from jina import Flow, Executor, requests

class MyGPUExec(Executor):
    def __init__(self, device: str = 'cpu', *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.device = device

    @requests
    def encode(self, docs: DocumentArray, **kwargs):
        with torch.inference_mode():
            # Generate random embeddings
            embeddings = torch.rand((len(docs), 5), device=self.device)
            docs.embeddings = embeddings
            embedding_device = 'GPU' if embeddings.is_cuda else 'CPU'
            docs.texts = [f'Embeddings calculated on {embedding_device}']
    
f = Flow().add(uses=MyGPUExec, uses_with={'device': 'cpu'})
docs = DocumentArray(Document())
with f:
  docs = f.post(on='/encode', inputs=docs)
print(f'Document embedding: {docs.embeddings}')
print(docs.texts)
