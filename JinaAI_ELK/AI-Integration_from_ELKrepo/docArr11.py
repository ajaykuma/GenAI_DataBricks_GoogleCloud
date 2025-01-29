#To process using external Flow/Sandbox/Executor to “process” a DocumentArray via post().
# The external Flow/Executor can be local, remote, or inside a Docker container.
from docarray import DocumentArray,Document
da = DocumentArray.empty(10)
r = da.post('grpc://ip:port')
r.summary()

#Using executor from executor hub
da = DocumentArray([Document(text='This is a test')])
r = da.post('jinahub+sandbox://xxxxxxx', show_progress=True)
r.summary()

#single document processing
d = Document(text='This is a test')
r = d.post('jinahub+sandbox://xxxxxx')

#post() accepts a URI-like scheme, supporting a wide range of Flows/Hub Executors:



