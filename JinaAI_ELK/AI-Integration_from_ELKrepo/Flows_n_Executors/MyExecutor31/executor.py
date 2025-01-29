'''
manually construct a nested Document, for example to hold different modalities, like text and image.
To construct multimodal Documents in a more comfortabe, readable, and idiomatic way you should use DocArray’s dataclass API.
'''
#Documents can be nested inside .chunks and .matches
from docarray import Document

d = Document(
    id='d0',
    chunks=[Document(id='d1', chunks=Document(id='d2'))],
    matches=[Document(id='d3')],
)

print(d)
print(d.summary())
print(d.granularity)
print(d.chunks)

#giving an unknown attribute
#If you give an unknown attribute (i.e. not one of the built-in Document attributes), it is automatically “caught” into the .tags attribute
d = Document(hello='world')
print(d, d.tags)

#esolve external fields into built-in attributes by specifying a mapping in field_resolver.
d = Document(hello='world', field_resolver={'hello': 'id'})
print(d)
print(d.non_empty_fields)
print(d.id)

#copy from another document
d = Document(text='hello')
d1 = Document(d, copy=True)
print(d == d1, id(d) == id(d1))
#This indicates d and d1 have identical content, but they are different objects in memory.

#keep the memory address of a Document object while only copying the content from another Document, you can use copy_from().
d1 = Document(text='hello')
d2 = Document(text='world')

print(id(d1))
d1.copy_from(d2)
print(d1.text)
print(id(d1))
