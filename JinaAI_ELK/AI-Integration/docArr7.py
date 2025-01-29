#working with DocumentArray
from docarray import Document,DocumentArray
da = DocumentArray()
da.append(Document(text='hello world!'))
da.extend([Document(text='hello'), Document(text='world!')])
da.summary()
print(da[0].content,da[1].content)
print(da[0].id,da[0].content,"\n",da[1].id,da[1].content)
print(da['be47bbdcc3d0d236f353940071900fb7'])
print(da['be47bbdcc3d0d236f353940071900fb7'].content)

newdoc = Document(text='test world!')
da.extend([newdoc])
da.summary()
newdoc in da

da.__len__()

#using boolean mask for updating or filtering certain Documents
mask = [True, False] * 2
del da[mask]
print(da)
print(da.__len__())

#Working with nested structures
#DocumentArray provides makes it easy to traverse over the nested structure and select Documents:
#A path represents the route from the top-level Documents to the destination. Use c to select chunks,
# cc to select chunks of chunks, m to select matches, mc to select chunks of matches, r to select top-level Documents.

#using .empty to add empty documents
da = DocumentArray().empty(3)
print(da.summary())

#using .empty to add empty documents in chunks and matches
for d in da:
    d.chunks = DocumentArray.empty(2)
    d.matches = DocumentArray.empty(2)

da[0] = Document(text='Test')
da[0].content

for d in da:
    d.chunks = DocumentArray.empty(2)
    d.matches = DocumentArray.empty(2)

print(da.summary())

print(da['@m'])
print(da['@m'],"\n",da['@m'][0],"\n",da['@m'][1])
print(da['@c'],"\n",da['@c'][0],"\n",da['@c'][1])

print(da['@c,m'])
print(da['@c,m,r'])
#other options can be
# da['@mc']
# da['@cm,cc']
# da['@m,cm']
#da['@m:5,c:3]']

#flat DocumentArray without all nested structure
da1 = DocumentArray().empty(3)
for d in da1:
    d.chunks = DocumentArray.empty(2)
    d.matches = DocumentArray.empty(2)

da1[...].summary()

#Batching document array using
#match()
#map_batch() -- for in parallel, to overall cpu & gpu computation
da = DocumentArray.empty(1000)
for b_da in da.batch(batch_size=256):
    print(b_da)

#using sampling
da = DocumentArray.empty(1000).sample(10)
print(da)

#shuffling
da = DocumentArray.empty(1000)
da.shuffle()

#Splitting by tags
#We can split a DocumentArray into multiple DocumentArrays according to a tag value
# (stored in tags) of each Document. It returns a Python dict where Documents with the same tag value are grouped
# together in a new DocumentArray, with their orders preserved from the original DocumentArray.

da = DocumentArray(
    [
        Document(tags={'category': 'c'}),
        Document(tags={'category': 'c'}),
        Document(tags={'category': 'b'}),
        Document(tags={'category': 'a'}),
        Document(tags={'category': 'a'}),
    ]
)

rv = da.split_by_tag(tag='category')
da.summary()








