from jina import Flow

from executor import SentenceEncoder


def generate_docs():
    for _ in range(10_000):
        yield Document(
            text='Using a GPU allows you to significantly speed up encoding.'
        )


f = Flow().add(uses=SentenceEncoder, uses_with={'device': 'cpu'})
with f:
    f.post(on='/encode', inputs=generate_docs, show_progress=True, request_size=32)

'''
f = Flow().add(
    uses='docker://sentence-encoder', uses_with={'device': 'cuda'}, gpus='all'
)
with f:
    f.post(on='/encode', inputs=generate_docs, show_progress=True, request_size=32)
'''
'''
we can add the above lines in main and then run
docker build -t sentence-encoder . 
'''
#then 'docker run sentence-encoder
