#Nested Structure
#Documents can be nested both horizontally and vertically via .matches and .chunks
from docarray import Document

d = Document(chunks=[Document(text='Hello')], matches=[Document(text='Hi'), Document(text='Hey')])
print(d.non_empty_fields)
print(d)
print(d.summary())
print(d.chunks.contents)
print(d.matches.contents)

d.chunks = [Document(text='World'), Document(text='Countries')]
print(d.chunks.contents)
print(d.matches.contents)
print(d.summary())
d.chunks.append(Document(text='Hello Again'))
print(d.summary())
print(d.chunks.contents)
print(d.matches.contents)
print(d)

print(type(d))
print(type(d.chunks))
print(type(d.matches))

#Visualize documents
import numpy as np
from docarray import Document,DocumentArray

d0 = Document(id='🐲', embedding=np.array([0, 0]))
d1 = Document(id='🐦', embedding=np.array([1, 0]))
d2 = Document(id='🐢', embedding=np.array([0, 1]))
d3 = Document(id='🐯', embedding=np.array([1, 1]))

d0.chunks.append(d1)
d0
d0.chunks
d0.chunks[0].chunks.append(d2)
d0.matches.append(d3)
print(d0.summary())

d0 = Document(text="world")
d1 = Document(text="country")
d2 = Document(text="city1")
d3 = Document(text="berlin")
d0.chunks.append(d1)
d0.chunks[0].chunks.append(d2)
d0.matches.append(d3)
da1 = DocumentArray([d0])
da1.summary()
da = DocumentArray([d0],
                   storage='elasticsearch',
                   config={'hosts': 'http://os1:9200', 'index_name': 'chunk_data','n_dim': 10},
                   )

#Using Fluent interface
#process (often preprocess) a Document object by chaining methods.
# For example to read an image file as numpy.ndarray, resize it, normalize it and then store it to another file

from docarray import Document

d = (
    Document(uri='/home/hdu/Downloads/apple1.jpg')
    .load_uri_to_image_tensor()
    .set_image_tensor_shape((64, 64))
    .set_image_tensor_normalization()
    .save_image_tensor_to_file('/home/hdu/Downloads/apple4.png')
)
