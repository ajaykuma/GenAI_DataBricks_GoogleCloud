#
import docarray
from docarray import DocumentArray,Document
#loading images from 'https://sites.google.com/view/totally-looks-like-dataset' or from Jina cloud
left_da = DocumentArray.pull('jina-ai/demo-leftda', show_progress=True)[:1000]
left_da.summary()
#left_da[0]
#to visualize--> (left_da.plot_image_sprites())
right_da = DocumentArray.pull('jina-ai/demo-rightda',show_progress=True)[:1000]
right_da.summary()

#computer vision pre-processing
def preproc(d: Document):
    return (
        d.load_uri_to_image_tensor()  # load
        .set_image_tensor_normalization()  # normalize color
        .set_image_tensor_channel_axis(-1, 0)
    )  # switch color axis for the PyTorch model later

left_da.apply(preproc)
right_da.apply(preproc)

#embedding images as done earlier
import torchvision

model = torchvision.models.resnet50(pretrained=True)  # load ResNet50
#left_da.embed(model, device='cuda')  # embed via GPU to speed up/or remove device
#error if using gpu on cpu
#Found no NVIDIA driver on your system. Please check that you have an NVIDIA GPU and installed a
# driver from http://www.nvidia.com/Download/index.aspx


left_da[0].embed(model)
right_da[0].embed(model)

#to visualize embeddings
#left_da.plot_embeddings(image_sprites=True)

#matching left and right
left_da.match(right_da, limit=9)

#what's inside left_da matches now
for m in left_da[0].matches:
    print(d.uri, m.uri, m.scores['cosine'].value)

#or using
print(left_da['@m', ('uri', 'scores__cosine__value')])

#we can also do a quantitative evaluation
#create groundtruth matches
groundtruth = DocumentArray(
    Document(uri=d.uri, matches=[Document(uri=d.uri.replace('left', 'right'))])
    for d in left_da
)

#Here we created a new DocumentArray with real matches by simply replacing the filename,
# e.g. left/00001.jpg to right/00001.jpg. T
# hat's all we need: if the predicted match has the identical uri as the groundtruth match,
# then it is correct

#check recall rate from 1 to 5 over the full dataset
for k in range(1, 6):
    print(
        f'recall@{k}',
        left_da.evaluate(
            groundtruth, hash_fn=lambda d: d.uri, metric='recall_at_k', k=k, max_rel=1
        ),
    )


#saving document array
#to binary, JSON, dict, DataFrame, CSV or Protobuf message
left_da.save('left_da.bin')

#to reuse
left_da = DocumentArray.load('left_da.bin')

#push or share
left_da.push('my_shared_da')

left_da = DocumentArray.pull('<username>/my_shared_da')
