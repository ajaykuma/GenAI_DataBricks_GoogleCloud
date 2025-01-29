#embedding
#a multi-dimensional representation of a Document (often a [1, D] vector)

import numpy as np
import scipy.sparse as sp
from docarray import Document,DocumentArray
import elasticsearch
from torchvision.models import ResNet50_Weights

d0 = Document(embedding=[1, 2, 3])
d1 = Document(embedding=np.array([1, 2, 3]))
d2 = Document(embedding=np.array([[1, 2, 3], [4, 5, 6]]))
d3 = Document(embedding=sp.coo_matrix([0, 0, 0, 1, 0]))

print(d0.embedding)
print(d0.non_empty_fields)
print(d0.embedding.count(2))
print(d2.summary())
print(d3.summary())

#instead of manually specifying embedding, use a DNN with .embed option
q = (Document(uri='/home/hdu/Downloads/apple1.jpg')
     .load_uri_to_image_tensor()
     .set_image_tensor_normalization()
     .set_image_tensor_channel_axis(-1, 0))

#embed it into a vector
import torchvision
model = torchvision.models.resnet50(pretrained=True)
d = q.embed(model)

print(d.embedding)
print(d.summary())
print(d.non_empty_fields)
print(type(d))

#Documents with an .embedding can be “matched” against each other.
q1 = (Document(uri='/home/hdu/Downloads/apple1.jpg')
     .load_uri_to_image_tensor()
     .set_image_tensor_normalization()
     .set_image_tensor_channel_axis(-1, 0))

q2 = (Document(uri='/home/hdu/Downloads/apple2.png')
     .load_uri_to_image_tensor()
     .set_image_tensor_normalization()
     .set_image_tensor_channel_axis(-1, 0))

q3 = (Document(uri='/home/hdu/Downloads/apple3.png')
     .load_uri_to_image_tensor()
     .set_image_tensor_normalization()
     .set_image_tensor_channel_axis(-1, 0))

print(q1.summary())
print(q2.summary())
print(q3.summary())

import torchvision
model = torchvision.models.resnet50(pretrained=True)
d1 = q1.embed(model)
d2 = q2.embed(model)
d3 = q3.embed(model)

from docarray import DocumentArray
da = DocumentArray([d1,d2,d3])
print(da)
print(da.contents)
print(da.embeddings.shape)
print(da.summary)
da.
#print(da.embeddings)
d1.match(da)
d2.match(da)
d3.match(da)
da.count(d1)

for doc in da:
    print(da.index(doc),doc.id)

#we create ten Documents and put them into a DocumentArray, and then use another Document to search against them.
da = DocumentArray.empty(10)
da.embeddings = np.random.random([10, 256])
q = Document(embedding=np.random.random([256]))
da.summary()
for i in da:
     print(i,da.index(i))
q.match(da)
q.summary()

################
#using feature hashing for embedding and distance metric is 'jaccard distance'
#--https://en.wikipedia.org/wiki/Feature_hashing
#--https://en.wikipedia.org/wiki/Jaccard_index
#searching for top 5 similar sentences
from docarray import Document, DocumentArray

d = Document(uri='https://www.gutenberg.org/files/1342/1342-0.txt').load_uri_to_text()
da = DocumentArray(Document(text=s.strip()) for s in d.text.split('\n') if s.strip())
da.apply(Document.embed_feature_hashing, backend='process')

q = (
    Document(text='she is')
    .embed_feature_hashing()
    .match(da, metric='jaccard', use_scipy=True)
)

print(q.matches[:5, ('text', 'scores__jaccard__value')])

#If using document array to be stored in Elasticsearch and then continuing as done above/searching in ES
d = Document(uri='https://www.gutenberg.org/files/1342/1342-0.txt').load_uri_to_text()
da = DocumentArray(Document(text=s.strip()) for s in d.text.split('\n') if s.strip())
#If storing in ES
#before storing document array in index, we can create an index to have controlled mappings for
#--fields such as text which will contain our selected text , we can make it i.e the field searchable
#--we can decide to have dense_vector which will contain embeddings to say 10 and specify same as n_dim

#create index with mappings..
# PUT /gutenberg_data
# {
#      "mappings" : {
#       "dynamic" : "true",
#       "properties" : {
#         "blob" : {
#           "type" : "text",
#           "fields" : {
#             "keyword" : {
#               "type" : "keyword",
#               "ignore_above" : 256
#             }
#           }
#         },
#         "embedding" : {
#           "type" : "dense_vector",
#           "dims" : 10
#         },
#         "text" : {
#           "type" : "text",
#           "index" : true
#         }
#       }
#     }
#
# ,
#   "settings": {
#     "number_of_shards": 2
#     , "number_of_replicas": 0
#     , "auto_expand_replicas": false
#   }
# }

#check mappings
#GET /gutenberg_data/_mapping
#now we can have this run
da = DocumentArray((Document(text=s.strip()) for s in d.text.split('\n') if s.strip()),
                   storage='elasticsearch',
                   config={'hosts': 'http://os1:9200', 'index_name': 'gutenberg_data','n_dim': 10},
                   )
#rest steps of applying model and doing a match can be done same as mentioned above.
#after index is created in ES
# GET _cat/indices?v
# GET _cat/shards/gutenberg_data?v
# POST / _sql?format = txt
# {
#
#      "query": "select text from gutenberg_data ", "fetch_size": 15}
#
# GET / gutenberg_data / _search
# {
#      "query": {
#           "match":
#                {"text": "masterly"}
#      }
# }
#
# --also check in Discover (text search-full or partial) after creating an index pattern




