#From/to JSON
#pip install "docarray[full]"
#Documents use JSON Schema and pydantic model for serialization, i.e. protocol='jsonschema'.
#When using a RESTful API, you should use protocol='jsonschema' as the resulting JSON will follow a pre-defined schema.
# To use Protobuf as the JSON serialization backend, pass protocol='protobuf' to the method
from docarray import Document
import numpy as np

d = Document(text='hello, world', embedding=np.array([1, 2, 3]))
d_as_json = d.to_json(protocol='protobuf')

print(d)
print(d,d_as_json)
print(d.embedding)
print(d.non_empty_fields)
print(d.id)
print(d.text)
print(d.content)
print(d.is_multimodal)

#From/to bytes
#Depending on your protocol and compress argument values, this feature may require protobuf and lz4 dependencies
'''Bytes or binary or buffer, however you want to call it, is probably 
the most common and compact wire format. DocArray provides to_bytes() and from_bytes() 
to serialize Document objects into bytes.'''
d = Document(text='hello, world', embedding=np.array([1, 2, 3]))
d_bytes = d.to_bytes()
d_r = Document.from_bytes(d_bytes)
print(d_bytes, "\n", d_r)
print(len(d_bytes))

#The default serialization protocol is pickle – you can change it to protobuf by specifying
# .to_bytes(protocol='protobuf'). You can also add compression to make the resulting bytes smaller
print(len(d.to_bytes(protocol='protobuf', compress='gzip')))

#with RESTful APIs, you can only send/receive strings, not bytes. You can serialize a Document
# into a base64 string with to_base64() and load it with from_base64()
d = Document(text='hello', embedding=[1, 2, 3])
print(d.to_base64())
print(len(d.to_base64()))
print(len(d.to_base64(protocol='protobuf', compress='lz4')))

#from/to dict
d_as_dict = Document(text='hello, world', embedding=np.array([1, 2, 3]))
d = d_as_dict.to_dict(protocol='protobuf')
print(type(d))
for i in d.items():
    print(i)
print(d)

#protobuf
d_proto = Document(uri='apple.jpg').to_protobuf()
print(type(d_proto), d_proto)
d = Document.from_protobuf(d_proto)









