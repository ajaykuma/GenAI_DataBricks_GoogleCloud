import docarray
import  numpy
import elasticsearch
docarray.__version__
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
da = DocumentArray([d1,d2,d3,d4],
    storage='elasticsearch',
    config={'hosts': 'http://os1:9200','index_name': 'new_stuff1','n_dim': 128},
)
da.summary()

