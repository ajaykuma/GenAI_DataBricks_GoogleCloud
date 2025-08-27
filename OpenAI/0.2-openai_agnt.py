##With OpenAI
#a LangChain Agent that translates text when instructed.
#Langchain Agent:
#!pip install langchain openai langchain_community langchain-core
from langchain.agents import initialize_agent, Tool
from langchain.llms import OpenAI
from langchain.tools import tool

# Define the Translation Tool
@tool
def translate_text(text: str, target_language: str) -> str:
    """Translates text into the target language."""
    translations = {
        "spanish": "Hola, ¿cómo estás?",
        "french": "Bonjour, comment ça va?",
        "german": "Hallo, wie geht es dir?"
    }
    return translations.get(target_language.lower(), "Translation not available.")

# Initialize OpenAI LLM
llm = OpenAI(model_name="gpt-4", temperature=0.7, openai_api_key="your_api_key")

# Create an agent with the translation tool
agent = initialize_agent(
    tools=[translate_text],
    llm=llm,
    agent="zero-shot-react-description",
    verbose=True
)

# Run the agent to translate text
response = agent.run("Translate 'Hello, how are you?' to Spanish.")
print(response)

##with AzureOpenAI
from langchain_openai import AzureChatOpenAI
from langchain.agents import initialize_agent

@tool
def translate_text(text: str, target_language: str) -> str:
    """Translates text into the target language."""
    translations = {
        "spanish": "Hola, ¿cómo estás?",
        "french": "Bonjour, comment ça va?",
        "german": "Hallo, wie geht es dir?"
    }
    return translations.get(target_language.lower(), "Translation not available.")

# Initialize Azure OpenAI LLM
llm = AzureChatOpenAI(
    openai_api_key="mykey",
    api_version="2023-12-01-preview",
    azure_endpoint="https://info-mdub1hyj-eastus2.cognitiveservices.azure.com/",
    deployment_name="gpt-4o-mini",  
    temperature=0.7,
)

agent = initialize_agent(
    tools=[translate_text],
    llm=llm,
    agent="zero-shot-react-description",
    verbose=True
)

# Run the agent to translate text
response = agent.run("Translate 'Hello, how are you?' to Spanish.")
print(response)
