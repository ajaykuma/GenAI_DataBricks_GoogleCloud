import docarray
import elasticsearch
import numpy
from docarray import DocumentArray, Document
from elasticsearch import Elasticsearch
d1 = Document(text='hello')
d2 = Document(blob=b'\f1')
d3 = Document(tensor=numpy.array([1, 2, 3]))
d4 = Document(
    uri='https://docarray.jina.ai',
    mime_type='text/plain',
    granularity=1,
    adjacency=3,
    tags={'foo': 'bar'},
)
d5 = Document(
    uri='https://docarray.jina.ai',
    mime_type='text/plain',
    granularity=1,
    adjacency=3
)

d6 = Document(
    dict(
        uri='https://docarray.jina.ai',
        mime_type='text/plain',
        granularity=1,
        adjacency=3,
    )
)

d7 = Document(
    {
      'uri': 'https://docarray.jina.ai',
        'mime_type': 'text/plain',
        'granularity': 1,
        'adjacency': 3,
    }
)

da2 = DocumentArray([d1,d2,d3,d4,d5,d6,d7],
    storage='elasticsearch',
    config={'hosts': 'http://os1:9200','index_name': 'new_stuff','n_dim': 128},
    )

# with da2:
#     da2.extend([Document() for _ in range(1000)])

da2.summary()