import tensorflow_hub as hub
import tensorflow_text as text
preprocess_url = 'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3'
encoder_url = 'https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4'
bert_preprocess_model = hub.KerasLayer(preprocess_url)
text_test = ['nide movie indeed', ' I love python programming']
text_preprocessed = bert_preprocess_model(text_test)
text_preprocessed.keys()
text_preprocessed['input_mask']
text_preprocessed['input_type_ids']
text_preprocessed['input_word_ids']
text_preprocessed['input_word_ids']
bert_model = hub.KerasLayer(encoder_url)
bert_results = bert_model(text_preprocessed)
bert_results.keys()
bert_results['pooled_output']
bert_results['sequence_output']
bert_results['encoder_outputs']


