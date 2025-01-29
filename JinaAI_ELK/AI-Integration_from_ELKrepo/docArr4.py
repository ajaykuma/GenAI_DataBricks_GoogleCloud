#Accessing attributes
#Use . expression to get/set the value of an attribute
import docarray
from docarray import  Document,DocumentArray
d = Document()
d.text = 'hello world'
print(d.text)
print(d.id)
print(d.non_empty_fields)
#unsetting an attribute
d.pop('text')
print(d.text)
d.text = 'hello world'
print(d.text)
d.text = None
print(d.text)

#the most important are content attributes, namely .text, .tensor, and .blob which contain the actual content.
#Each document can contain only one kind of attribute
import numpy as np
from docarray import Document

d = Document(text='hello')
print(d.text)
print(d)
d.tensor = np.array([1, 2, 3])
print(d)
print(d.tensor)
print(d.content)
print(d.non_empty_fields)

#to represent more than one kind of information i.e. text and images (such as coming from PDF), we can use
#nested documents
#Each Document contains only one modality of information.
d = Document(chunks=[Document(tensor=[1, 2, 3]), Document(text='Hello')])
print(d)
print(d.non_empty_fields)
print(d.chunks)
print(d.chunks.texts)


d.summary()


#using content setter and getter
d1 = Document(content='hello')
print(d1)
print(d1.content_type)
print(d1.content)

d1.content = [1,2,3]
print(d1)
print(d1.content_type)
print(d1.content)

#Load content from uri
d1 = Document(uri='/home/hdu/Downloads/apple2.png').load_uri_to_image_tensor()
#print(d1.content_type, d1.content)
print(d1.non_empty_fields)

d2 = Document(uri='https://www.gutenberg.org/files/1342/1342-0.txt').load_uri_to_text()
#print(d2.content_type, d2.content)










