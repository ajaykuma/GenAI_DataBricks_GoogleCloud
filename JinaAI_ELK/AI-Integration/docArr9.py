#Querying data
from pprint import pprint
from docarray import Document, DocumentArray

da = DocumentArray(
    [
        Document(
            text='magazine',
            weight=25,
            tags={'h': 14, 'w': 21, 'uom': 'cm'},
            modality='A',
        ),
        Document(
            text='notebook',
            weight=50,
            tags={'h': 8.5, 'w': 11, 'uom': 'in'},
            modality='A',
        ),
        Document(
            text='newspaper',
            weight=100,
            tags={'h': 8.5, 'w': 11, 'uom': 'in'},
            modality='D',
        ),
        Document(
            text='diary',
            weight=75,
            tags={'h': 22.85, 'w': 30, 'uom': 'cm'},
            modality='D',
        ),
        Document(
            text='postcard',
            weight=45,
            tags={'h': 10, 'w': 15.25, 'uom': 'cm'},
            modality='A',
        ),
    ]
)

da.summary()

#A query filter document uses query operators to specify conditions
#$eq,$ne,$gt,$gte,$lt,$lte,$in,$nin,$regex,$size,$exists
r = da.find({'modality': {'$eq': 'D'}})
type(r)
print(r.contents)
res = r.to_json(protocol="protobuf")
print(res)
pprint.pprint(res)
res = r.to_dict(exclude_none=True,protocol="protobuf")
pprint.pprint(res)

#searching for tags
r = da.find({'tags__h': {'$gt': 10}})
print(r.contents)

res = r.to_json(protocol="protobuf")
for i in res.split(","):
    print(i)
print(res)

#Using substitution with field
r = da.find({'tags__h': {'$gt': '{tags__w}'}})
print(r.contents)
res = r.to_json(protocol="protobuf")
print(res)

#Using multiple conditions
r = da.find({'$or': [{'weight': {'$eq': 45}}, {'modality': {'$eq': 'D'}}]})
print(r.contents)
res = r.to_json(protocol="protobuf")
print(res)


