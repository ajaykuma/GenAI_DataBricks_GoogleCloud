import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text

import pandas as pd
from keras.backend import shape

df = pd.read_csv("/home/hdu/Downloads/spamtrain.csv")
df.head(5)

#basic analysis
df.groupby('label').describe()
#should show you count of type of emails
#we need to handle inbalancing of dataset

#Let's use technique for 'down sampling '
df['label'].value_counts()

#check for percentage of emails in data

#discard some emails to have equal number of spam and ham emails
#we can also do oversampling of smaller data as other option

df_spam = df[df['label'] == 'spam']
df_spam.shape
df_spam.head()
df_ham = df[df['label'] == 'ham']
df_ham.shape
df_ham.head()

df_ham_downsampled = df_ham.sample(df_spam.shape[0])
df_ham_downsampled.shape

#concat both
df_balanced = pd.concat([df_spam, df_ham_downsampled])
df_balanced.columns
df_balanced['label'].value_counts()
#should show equal number of spam and ham emails

df_balanced.sample(5)

df_balanced['spam'] = df_balanced['label'].apply(lambda x: 1 if x == 'spam' else 0)
df_balanced.columns
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(df_balanced['label'], df_balanced['spam'],
                                                    stratify=df_balanced['spam'])
#Note** use startify so that in train and test sample, the distribution of categories is equal.

X_train.head
df_balanced.head()
preprocess_url = 'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3'
encoder_url = 'https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4'

bert_preprocess_model = hub.KerasLayer(preprocess_url)
bert_encoder = hub.KerasLayer(encoder_url)

#let's write a function to get embedding for a sentence

def get_sentence_embedding(sentences):
    preprocessed_text = bert_preprocess_model(sentences)
    return bert_encoder(preprocessed_text)['pooled_output']

get_sentence_embedding([
    "100$ discount, hurry up",
    "peter, are you available for a discussion tomorrow"])

#test bert encoder to get embeddings for some random words
e = get_sentence_embedding([
    "banana",
    "grapes",
    "mango",
    "jeff bezos",
    "elon musk",
    "bill gates"
])

from sklearn.metrics.pairwise import cosine_similarity

# we use consine similarity to see how similar two vectors are,
# if cosine simlarity is close to 1 that means vectors are similar

cosine_similarity([e[0]], [e[1]])

#similarly
cosine_similarity([e[0]], [e[3]])

# we can create sequential or functional model in Keras

# Bert layers
# create input layer
text_input = tf.keras.layers.Input(shape=(),dtype=tf.string, name="label")
preprocessed_text = bert_preprocess_model(text_input)
outputs = bert_encoder(preprocessed_text)

# Neural network layers
# dropout layer : refers to dropping out the nodes (input and hidden layer) in a neural network
l = tf.keras.layers.Dropout(0.1, name='dropout')(outputs['pooled_output'])
l = tf.keras.layers.Dense(1, activation='sigmoid', name='output')(l)
#Note ** if more than 0.5 sigmoid = spam or not

# construct final model
model = tf.keras.Model(inputs=[text_input], outputs=[l])

model.summary()

#should show you 'Trainable params': 769(input: 768 + 1)
#remaining params come from BERT which are not trainable

METRICS = [
    tf.keras.metrics.BinaryAccuracy(name='accuracy'),
    tf.keras.metrics.Precision(name='precision'),
    tf.keras.metrics.Recall(name='recall')]

model.compile(optimizer='adam',
              loss='binary_crossentropy',
                metrics = METRICS)

model.fit(X_train, y_train, epochs=10)
# The number of epochs is a hyperparameter that defines the number times that the learning algorithm will work through the entire training dataset.

model.evaluate(X_test, y_test)

y_predicted = model.predict(X_test)
y_predicted = y_predicted.flatten()

import numpy as np

y_predicted = np.where(y_predicted > 0.5, 1, 0)
y_predicted

from sklearn.metrics import confusion_matrix, classification_report

cm = confusion_matrix(y_test, y_predicted)
cm

from matplotlib import pyplot as plt
import seaborn as sn

sn.heatmap(cm, annot=True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('Truth')

print(classification_report(y_test, y_predicted))
#should show good precision, recall and fl_score

reviews = [
    'Enter  a chance to win $5000, hurry up, offer valid until march 31,2011',
    'You are awarded a SiPix Digital Camera! call 00191928267 from landline. Delivery within 27days.',
    'it is 80488. Your 500 free text messages are valid until 31 December 2005',
    'Hey peter, are you coming to office today',
    "why don't you wait for 1 week to check your salary"]

model.predict(reviews)
# Look at values and see if its more than 0.5 to be categorized as SPAM.
