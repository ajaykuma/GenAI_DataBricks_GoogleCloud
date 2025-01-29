#Parallelization
# map(): to parallel process Document by Document, return an interator of elements;
# map_batch(): to parallel process minibatch DocumentArray, return an iterator of DocumentArray;
# apply(): like .map(), modify a DocumentArray inplace;
# apply_batch(): like .map_batch(), modify a DocumentArray inplace.

#map() maps function to every element of the DocumentArray in parallel. (map_batch() for mini batch)
from docarray import DocumentArray
#image Documents with .uri set
docs = DocumentArray.from_files('/home/hdu/Downloads/*.png')
print(docs[0])
print(docs[0].uri)

for i in docs:
    print(i.uri)

print(type(docs))
docs.summary()

#load and preprocess the Documents
def foo(d):
    return (
        d.load_uri_to_image_tensor()
         .set_image_tensor_shape((64, 64))
         .set_image_tensor_normalization()
    )

#By default, parallelization # is conducted with thread backend, i.e. multi-threading.
# It also supports process backend by setting .apply(..., backend='process').

#For loop
for d in docs:
    foo(d)

#Apply in parallel
docs.apply(foo)

#pip install pyqt5
docs.plot_image_sprites()

#to see matching
docs[0].plot_matches_sprites(top_k=5, channel_axis=-1, inv_normalize=False)

#visualizing using embeddings
import numpy as np
docs = DocumentArray.empty(1000)
docs.embeddings = np.random.random([len(docs), 256])
docs.plot_embeddings()
docs.plot_embeddings(image_sprites=True)




