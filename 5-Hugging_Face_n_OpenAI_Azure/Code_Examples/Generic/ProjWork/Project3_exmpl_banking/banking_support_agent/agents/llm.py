"""
Shared LLM instance used by all agents.

Centralizing this means switching model/provider (e.g. to Claude via
Anthropic, or to a local Ollama model) is a one-line change here instead of
edits scattered across every agent file.
"""

from crewai import LLM
#from langchain_openai import AzureChatOpenAI
from config import (
    azure_endpoint,
    api_key,
    api_version,
    azure_deployment,
)

# print("endpoint:", repr(azure_endpoint))
# print("deployment:", repr(azure_deployment))
# print("api_version:", repr(api_version))
# print("api_key set:", bool(api_key))

# def get_llm() -> AzureChatOpenAI:
#     return AzureChatOpenAI(
#         azure_endpoint=azure_endpoint,
#         api_key=api_key,
#         api_version=api_version,
#         deployment_name=azure_deployment,
#         temperature=0.5,
#     )

def get_llm() -> LLM:
    return LLM(
        model=f"azure/{azure_deployment}",
        api_key=api_key,
        api_base=azure_endpoint,
        api_version=api_version,
        temperature=0.5,
    )

# llm = get_llm()
# resp = llm.call("Say hello in one word.")
# print(resp)
if __name__ == "__main__":
    llm = get_llm()
    resp = llm.call("Say hello in one word.")
    print(resp)
