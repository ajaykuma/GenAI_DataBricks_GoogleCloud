#Creating separate llmchain instances for different models:

from langchain.chains import LLMChain
from langchain import HuggingFaceHub
from langchain.prompts import PromptTemplate

# Define prompt
template = "Question: {question}\nAnswer: Let's think step by step."
prompt = PromptTemplate(template=template, input_variables=["question"])

# Load two different models
llm_1 = HuggingFaceHub(repo_id="google/flan-t5-large", model_kwargs={"temperature": 0.7, "max_length": 512})
llm_2 = HuggingFaceHub(repo_id="tiiuae/falcon-7b-instruct", model_kwargs={"temperature": 0.7, "max_length": 512})

# Create two separate chains
llm_chain_1 = LLMChain(prompt=prompt, llm=llm_1)
llm_chain_2 = LLMChain(prompt=prompt, llm=llm_2)

# Use a model based on user choice
question = "What is quantum mechanics?"
model_choice = "flan-t5"  # Example of selecting a model dynamically

if model_choice == "flan-t5":
    response = llm_chain_1.run(question)
else:
    response = llm_chain_2.run(question)

print(response)

#Using a Function to Dynamically Select the Model
def get_llm_chain(model_name):
    model_map = {
        "flan-t5": HuggingFaceHub(repo_id="google/flan-t5-large"),
        "falcon-7b": HuggingFaceHub(repo_id="tiiuae/falcon-7b-instruct")
    }
    
    llm = model_map.get(model_name)
    if not llm:
        raise ValueError("Invalid model name")

    return LLMChain(prompt=prompt, llm=llm)

# Example usage
llm_chain = get_llm_chain("flan-t5")
response = llm_chain.run("Explain relativity in simple terms.")
print(response)



