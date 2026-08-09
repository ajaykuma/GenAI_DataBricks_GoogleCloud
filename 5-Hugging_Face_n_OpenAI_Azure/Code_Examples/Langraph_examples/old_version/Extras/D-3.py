#With(& If llama3 deployed locally)
import streamlit as st
from pydantic import BaseModel
from langchain_community.llms import Ollama

# Initialize the Ollama model
llm = Ollama(model="llama3")  # You can use mistral, gemma, codellama, etc.

# Define a structured schema using Pydantic (for clarity)
class WebSearchPrompt(BaseModel):
    search_query: str
    justification: str

# Streamlit UI
st.title("Web Search Optimization with Ollama")
st.write("Enter a question to receive an optimized web search query and reasoning.")

user_query = st.text_input("Enter your question:")

if user_query:
    # Prompt engineering to simulate structure
    prompt = (
        f"You are an intelligent assistant. Generate an optimized web search query for this question:\n"
        f"'{user_query}'\n"
        f"Then, explain why that search query is suitable.\n\n"
        f"Respond in the following format:\n"
        f"Search Query: <query>\nJustification: <reasoning>"
    )

    # Get response from local model
    response = llm.invoke(prompt)

    # Attempt to parse structured response
    try:
        lines = response.splitlines()
        search_query = next((line.split(":", 1)[1].strip() for line in lines if line.lower().startswith("search query")), "")
        justification = next((line.split(":", 1)[1].strip() for line in lines if line.lower().startswith("justification")), "")

        result = WebSearchPrompt(search_query=search_query, justification=justification)

        st.subheader("Optimized Search Query:")
        st.write(result.search_query)

        st.subheader("Reasoning:")
        st.write(result.justification)

    except Exception as e:
        st.error(f"Failed to parse response: {e}")
        st.write("Raw response:")
        st.text(response)
