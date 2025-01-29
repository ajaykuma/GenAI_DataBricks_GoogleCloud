from docarray import Document
import numpy

#construct an empty document
d = Document()
print(d.id)

#Each Document has a unique random id to identify it. It can be used to access the Document inside a DocumentArray.
#The random id is the hex value of UUID1. To convert it into the a UUID string

import uuid
dconv = str(uuid.UUID(d.id))

#initializing a Document object with the given attributes
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

#Use print that shows the Document’s non-empty attributes as well as its id
print('d is   :', d)
print(type(d))
print('dconverted is    :' ,dconv)
print('d1 is    :', d1)
print(d1.text)
print(d1.tags)
print(d1.id)
print(d1.embedding)


print('d2 is    :', d2)
print(d2.blob)
print(d2.non_empty_fields)

print('d3 is    :', d3)
print(d3.content)
print(d3.embedding)
print('d4 is    :', d4,d4.tags)

#wrap keyword arguments into a dict
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

print('d5 is    :', d5)
print(d5.id)
print(d5.mime_type)
print(d5.is_multimodal)

print('d6 is    :', d6)
print('d7 is    :', d7)
print(d7.to_json(protocol="protobuf"))
print(d7.non_empty_fields)
print(d7.uri)
print(d7.summary())



