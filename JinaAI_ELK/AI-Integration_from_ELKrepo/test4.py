import docarray
#BaseDoc and DocLIst is replaced in newer version by Document and DocumentArray
#Here we are using docarray=0.21, change the version of docarray to latest to use this..
from docarray import BaseDoc, DocList
import numpy as np
from docarray.index import ElasticDocIndex  # or ElasticV7DocIndex
from pydantic import Field


class NewsDoc(BaseDoc):
    text: str
    category: str = Field(col_type='keyword')  # enable keyword filtering


doc_index = ElasticDocIndex[NewsDoc](index_name='my_index_new',hosts='http://os1:9200')
index_docs = [
    NewsDoc(id='0', text='this is a news for sport', category='sport'),
    NewsDoc(id='1', text='this is a news for finance', category='finance'),
    NewsDoc(id='2', text='this is another news for sport', category='sport'),
]
doc_index.index(index_docs)

# search with filer
query_filter = {'terms': {'category': ['sport']}}
docs = doc_index.filter(query_filter)
print(docs)
