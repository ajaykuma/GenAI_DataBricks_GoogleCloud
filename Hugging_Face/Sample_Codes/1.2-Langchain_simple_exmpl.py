!pip install transformers torch

import torch
from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer

# Load the model and tokenizer locally
model_name = "google/flan-t5-base"  # You can also use "google/flan-t5-xl"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Create a text generation pipeline
pipe = pipeline(
    "text2text-generation",
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.float32,  # Uses lower precision for efficiency
    device=0 if torch.cuda.is_available() else -1  # Use GPU if available
)

#from langchain_community.llms import HuggingFacePipeline --old version
!pip install -U langchain-huggingface
from langchain_huggingface import HuggingFacePipeline

llm = HuggingFacePipeline(pipeline=pipe)

#if not already imported earlier
from langchain_core.prompts import PromptTemplate
# Define prompt template
template = """Question: {question}
Answer: Let's think step by step."""
prompt = PromptTemplate(template=template, input_variables=["question"])

# Example questions
questions = [
    "Explain the concept of black holes in simple terms.",
    "What are the main causes of climate change, and how can we address them?",
    "Provide a brief overview of the history of artificial intelligence."
]

#Option 1(Older version)
#from langchain.chains import LLMChain
#Create LLMChain
#llm_chain = LLMChain(prompt=prompt, llm=llm)

# # Run model for each question
# for question in questions:
#     response = llm_chain.run(question)
#     print(f"Q: {question}\nA: {response}\n")

#Option 2(newer version)
from langchain_core.runnables import RunnableSequence
# # RunnableSequence
chain = prompt | llm

for q in questions:
     print(f"\nQ: {q}")
     print(chain.invoke({"question": q}))

