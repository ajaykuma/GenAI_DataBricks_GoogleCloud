#BaseDoc and DocLIst is replaced in newer version by Document and DocumentArray
#Here we are using docarray=0.21, change the version of docarray to latest to use this..
import docarray
from docarray import BaseDoc, DocList
import numpy as np
import elasticsearch7
from docarray.index import ElasticDocIndex  # or ElasticV7DocIndex
import numpy

# Define the document schema.
class MyDoc(BaseDoc):
    title: str

    # Create dummy documents.
docs = DocList[MyDoc](MyDoc(title=f'title #{i}', embedding=np.random.rand(128)) for i in range(10))
#
# # Initialize a new ElasticDocIndex instance and add the documents to the index.
doc_index = ElasticDocIndex[MyDoc](index_name='my_index',hosts='http://os1:9200')
print(type(doc_index))
doc_index.index(docs)


#check index from elasticsearch
