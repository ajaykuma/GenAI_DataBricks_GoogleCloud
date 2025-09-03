#This code bypasses Hugging Face’s API and runs the model locally (may require GPU).
#Instead of HuggingFaceHub, Running by loading the model locally
#no api keys required here

from transformers import pipeline
#from langchain.llms import HuggingFacePipeline <--deprecated
#from langchain_community.llms import HuggingFacePipeline

pipe = pipeline("text2text-generation", model="google/flan-t5-large")
# response = pipe("Explain the concept of black holes in simple terms.")
# print('first output: ',response[0]['generated_text'])

questions = [
    "Explain the concept of black holes in simple terms.",
    "What are the main causes of climate change, and how can we address them?",
    "Provide a brief overview of the history of artificial intelligence."
]

# for question in questions:
#     response = pipe(question)
#     print(f"Q: {question}\nA: {response[0]['generated_text']}\n")

# Run model for each question
for question in questions:
    response = pipe(question)
    print(f"Q: {question}\nA: {response}\n")

# for question in questions:
#     response = pipe(question)
#     print(f"Q: {question}\nA: {response[0]['generated_text']}\n")
