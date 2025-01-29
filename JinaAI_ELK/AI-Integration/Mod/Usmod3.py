import pandas as pd
df = pd.read_csv('/home/hdu/Downloads/spamtrain.csv')
df.head()
df.columns

#independent features
X = list(df['text'])
X
#dependent features
y = list(df['label'])
y

y = list(pd.get_dummies(y,drop_first=True)['spam'])
y

from sklearn.model_selection import train_test_split
X_train,X_test, y_train,y_test = train_test_split(X,y,test_size=0.20, random_state=0)

#pip install transformers
#call the pretrained model > call the tokenizer >
from transformers import DistilBertTokenizerFast
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

#padding to make all sentences of same size
#truncation to remove white spaces
train_encodings = tokenizer(X_train, truncation=True, padding=True)
test_encodings = tokenizer(X_test, truncation=True, padding=True)

train_encodings
test_encodings

#convert encodings into dataset objects
import tensorflow as tf
train_dataset = tf.data.Dataset.from_tensor_slices((dict(train_encodings),y_train))
test_dataset = tf.data.Dataset.from_tensor_slices((dict(test_encodings),y_test))

from transformers import TFDistilBertForSequenceClassification, TFTrainer, TFTrainingArguments

training_args = TFTrainingArguments(
    output_dir = '/home/hdu/results',   #output dir
    num_train_epochs=2,        #total number of training epochs
    per_device_train_batch_size=8, #batch size per device during training
    per_device_eval_batch_size=16,  #batch size for evaluation
    warmup_steps=500,                #number of warmup steps for learning rate scheduler
    weight_decay=0.01,               #strength of weight decay
    logging_dir='/home/hdu/logs',           #dirctory for storing logs
    logging_steps=10,
)

with training_args.strategy.scope():
    model = TFDistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased")

trainer = TFTrainer(
    model=model,         #the instantiated transformers model to be trained
    args=training_args,     #training arguments defined above
    train_dataset=train_dataset,                 #training_dataset
    eval_dataset=test_dataset,                   #evaluation dataset
)

trainer.train()

trainer.evaluate(test_dataset)

trainer.evaluate(test_dataset)
trainer.predict(test_dataset)
trainer.predict(test_dataset)[1].shape

output = trainer.evaluate(test_dataset)[1]

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,output)

cm

trainer.save_model('senti_model')


