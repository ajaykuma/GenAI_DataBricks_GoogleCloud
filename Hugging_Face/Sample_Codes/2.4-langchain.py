#Creating separate llmchain instances for different models:
import torch
from transformers import pipeline
from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.llms import HuggingFacePipeline

# Define prompt
template = "Question: {question}\nAnswer: Let's think step by step."
prompt = PromptTemplate(template=template, input_variables=["question"])

# Load the model and tokenizer locally
model1_name = "google/flan-t5-large"  # You can also use "google/flan-t5-xl"
model2_name = "tiiuae/falcon-7b-instruct"

# Create a text generation pipeline
pipe1 = pipeline(
    "text2text-generation",
    model=model1_name,
    torch_dtype=torch.float32,  # Uses lower precision for efficiency
    device=0 if torch.cuda.is_available() else -1  # Use GPU if available
)

pipe2 = pipeline(
    "text-generation",
    model=model2_name,
    torch_dtype=torch.float32,  # Uses lower precision for efficiency
    device=0 if torch.cuda.is_available() else -1  # Use GPU if available
)

llm1 = HuggingFacePipeline(pipeline=pipe1)
llm2= HuggingFacePipeline(pipeline=pipe2)

# Create two separate chains
llm_chain_1 = LLMChain(prompt=prompt, llm=llm1)
llm_chain_2 = LLMChain(prompt=prompt, llm=llm2)

# Use a model based on user choice
question = "What is quantum mechanics?"
model_choice = "flan-t5"  # Example of selecting a model dynamically

if model_choice == "flan-t5":
    response = llm_chain_1.run(question)
else:
    response = llm_chain_2.run(question)

print(response)

#Using a Function to Dynamically Select the Model (uncmment below code)
# def get_llm_chain(model_name):
#     model_map = {
#         "flan-t5": model1_name,
#         "falcon-7b": model2_name
#     }
    
#     llm = model_map.get(model_name)
#     if not llm:
#         raise ValueError("Invalid model name")

#     return LLMChain(prompt=prompt, llm=llm)

# # Example usage
# llm_chain = get_llm_chain("flan-t5")
# response = llm_chain.run("Explain relativity in simple terms.")
# print(response)



