import docarray
import elasticsearch
from docarray import DocumentArray, Document
from elasticsearch import Elasticsearch


da2 = DocumentArray(
    storage='elasticsearch',
    config={'hosts': 'http://os1:9200','index_name': 'old_stuff','n_dim': 128},)

with da2:
    da2.extend([Document() for _ in range(1000)])

da2.summary()