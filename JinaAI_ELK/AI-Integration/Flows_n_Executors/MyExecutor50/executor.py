from docarray import dataclass, Document
from docarray.typing import Image, Text, JSON

@dataclass
class mmdocexample:
    banner: Image
    headline: Text
    meta: JSON


a = mmdocexample(
    banner='/home/hdu/Downloads/apple2.png',
    headline='Simple pic of a fruit, ...',
    meta={
        'author': 'Someone imp',
        'Column': 'By the Way - A healthy fruit',
    },
)

d = Document(a)
print(d)
print(d.is_multimodal)
print(d.non_empty_fields)
print(d.summary)
print(d._metadata)
#to see all details
print(d.chunks.to_json())
