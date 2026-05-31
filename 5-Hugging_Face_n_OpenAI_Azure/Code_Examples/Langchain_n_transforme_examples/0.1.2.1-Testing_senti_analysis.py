# Install dependencies if not already installed
# pip install transformers torch

from transformers import pipeline

# # Step 1: Load a pre-trained sentiment-analysis pipeline
# sentiment_analyzer = pipeline("sentiment-analysis")

# # Step 2: Input sentence
# sentence = "am feeling very happy"

# # Step 3: Analyze sentiment
# result = sentiment_analyzer(sentence)

# # Step 4: Show the result
# print(result)

# sentence = "am feeling ok but very scared"

# # Step 3: Analyze sentiment
# result = sentiment_analyzer(sentence)

# # Step 4: Show the result
# print(result)

# Install dependencies if not already installed
# pip install transformers torch

from transformers import BertTokenizer, BertForSequenceClassification
import torch
import torch.nn.functional as F

# Step 1: Load pre-trained BERT model fine-tuned for sentiment analysis
model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name)

# Step 2: Input sentence
sentence = "am feeling very happy"

# Step 3: Tokenize input (NLP)
inputs = tokenizer(sentence, return_tensors="pt")  # converts text to token IDs

#print(inputs)

# Step 4: Get model predictions (NLU)
with torch.no_grad():
    outputs = model(**inputs)

# Step 5: Convert logits to probabilities
probs = F.softmax(outputs.logits, dim=1)

# Step 6: Get predicted class
pred_class = torch.argmax(probs, dim=1).item()
confidence = probs[0][pred_class].item()

# Step 7: Map class index to sentiment label
# nlptown/bert-base-multilingual-uncased-sentiment outputs 1-5 stars
labels = {
    0: "Very Negative",
    1: "Negative",
    2: "Neutral",
    3: "Positive",
    4: "Very Positive"
}

print(f"Predicted sentiment: {labels[pred_class]}, Confidence: {confidence:.4f}")

