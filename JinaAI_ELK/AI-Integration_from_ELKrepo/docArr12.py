#Embeddings
#When DocumentArray has .tensors set, you can use a neural network to embed()
# it into vector representations, i.e. filling .embeddings.

from docarray import DocumentArray
import numpy as np
docs = DocumentArray.empty(10)
docs.tensors = np.random.random([10, 128]).astype(np.float32)

#use a simple MLP in PyTorch for model
import torch

model = torch.nn.Sequential(
    torch.nn.Linear(
        in_features=128,
        out_features=128,
    ),
    torch.nn.ReLU(),
    torch.nn.Linear(in_features=128, out_features=32))

docs.embed(model)
print(docs.embeddings)

#By default, the filled .embeddings are in the given model framework’s format.
# If you want them to always be numpy.ndarray, use .embed(..., to_numpy=True).

#You can specify .embed(..., device='cuda') when working with a GPU.
# The device name identifier depends on the model framework that you’re using.

#On large DocumentArrays that don’t fit into GPU memory,
# you can set batch_size with .embed(..., batch_size=128).

#use a pretrained model from PyTorch
import torchvision
model = torchvision.models.resnet50(pretrained=True)
docs.embed(model)
print(docs.embeddings)

#Note that .embed() only works when you have .tensors set,
# if you have .texts set and your model function supports strings as input,
#da = DocumentArray(...)
#da.embeddings = my_text_model(da.texts)
