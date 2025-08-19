import torch
from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer
#from langchain.llms import HuggingFacePipeline <--deprecated
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

# Load the model and tokenizer locally
model_name = "google/flan-t5-large"  # You can also use "google/flan-t5-xl"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Create a text generation pipeline
pipe = pipeline(
    "text2text-generation",
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.float16,  # Uses lower precision for efficiency
    device=0 if torch.cuda.is_available() else -1  # Use GPU if available
)

generation_kwargs = {
    "temperature": 0.7,   # Controls randomness (higher = more creative)
    #"max_length": 512,    # Max number of tokens in response
    "max_new_tokens": 20, # max tokens
    #"min_length": 150,      # Forces at least 150 words (~150 tokens)
    "top_p": 0.9,         # Nucleus sampling (higher = more diverse responses)
    "top_k": 50,          # Limits the number of top tokens considered
    "repetition_penalty": 1.2,  # Penalizes repetition (1.0 = no penalty)
    "do_sample": True,    # Enables sampling (for creative responses)
}

# Wrap pipeline in LangChain's HuggingFacePipeline with parameters
llm = HuggingFacePipeline(pipeline=pipe, model_kwargs=generation_kwargs)

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

# Create LLMChain
#Option 1(Older version)
# Create LLMChain
#llm_chain = LLMChain(prompt=prompt, llm=llm)

# # Run model for each question
# for question in questions:
#     response = llm_chain.run(question)
#     print(f"Q: {question}\nA: {response}\n")

#Option 2(newer version)
# # RunnableSequence
chain = prompt | llm

for q in questions:
     print(f"\nQ: {q}")
     print(chain.invoke({"question": q}))
