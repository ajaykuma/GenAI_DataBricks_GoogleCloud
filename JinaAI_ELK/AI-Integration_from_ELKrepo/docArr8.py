#A DocumentArray itself has no attributes. Accessing attributes in this context
# means accessing attributes of the contained Documents in bulk.

#da[element_selector, attribute_selector]
import numpy as np
from docarray import DocumentArray, Document

da = DocumentArray().empty(3)
for d in da:
    d.chunks = DocumentArray.empty(2)
    d.matches = DocumentArray.empty(2)

da.summary()
print(da[:, 'id'])
print(da['@c', 'id'])
print(da[..., 'id'])

#set mime_type for all top-level documents
da[:, 'mime_type'] = ['image/jpg', 'image/png', 'image/jpg']
da.summary()

#deleting
del da[:, 'mime_type']
da.summary()

#Attributes like .tags and .scores are nested by nature
# Accessing the deep nested value is easy using the dunder expression.
# Access .tags['key1'] via d[:, 'tags__key1']

da = DocumentArray.empty(3)
da.embeddings = np.random.random([3, 2])
da.match(da)

print(da['@m', ('id','scores__cosine__value')])
print(da.embeddings)

d = Document(hello='world')
print(d, d.tags)
da.append(d)
da.summary()
print(da[:, 'tags__hello'])

#We can use .texts, .blobs, .tensors, .contents and .embeddings
# attributes for quickly accessing the content and embeddings of Documents.

# build sparse matrix
embed = np.random.random([3, 10])
da = DocumentArray.empty(3)
da.embeddings = embed

print(type(da.embeddings), da.embeddings.shape)
for d in da:
    print(type(d.embedding), d.embedding.shape)



