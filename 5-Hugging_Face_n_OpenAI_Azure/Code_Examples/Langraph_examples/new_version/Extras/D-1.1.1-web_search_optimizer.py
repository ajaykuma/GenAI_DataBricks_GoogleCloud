'''Run it using > streamlit run D-1.1.py'''
import os
from typing import List

import streamlit as st
from pydantic import BaseModel, Field
from langchain_openai.chat_models import AzureChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv

load_dotenv()

client = AzureChatOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("API_KEY"),
    api_version="2025-01-01-preview",
    deployment_name="gpt-4.1",
    temperature=0,
)


# --------------------------------------------------------------------
# Schema matches everything the prompt actually asks for, so the
# model has a labeled slot for each piece.
# --------------------------------------------------------------------
class WebSearchPrompt(BaseModel):
    search_query: str = Field(description="A concise, optimized web search query")
    one_line_answer: str = Field(description="A single-sentence direct answer to the user's question")
    justification: str = Field(description="Why this search query was chosen, in 1-2 sentences")
    alternative_queries: List[str] = Field(
        default_factory=list,
        description="2-3 alternative search queries the user could try",
    )


# with_structured_output binds the schema via the model's native
# JSON/tool-calling mode -- no markdown-fence stripping, no manual
# json.loads(), no risk of the model wrapping output in ```json.
structured_client = client.with_structured_output(WebSearchPrompt)

SYSTEM_PROMPT = (
    "You turn a user's question into an optimized web search query. "
    "Also give a short direct answer from your own knowledge, a brief "
    "justification for the query you chose, and a few alternative queries "
    "the user could try if the first one doesn't return good results."
)

st.title("Web Search Optimization with LLM")
user_query = st.text_input("Enter your question:")

if user_query:
    try:
        structured: WebSearchPrompt = structured_client.invoke(
            [
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=f'User question: "{user_query}"'),
            ]
        )

        st.subheader("Optimized Search Query")
        st.write(structured.search_query)

        st.subheader("Quick Answer")
        st.write(structured.one_line_answer)

        st.subheader("Reasoning")
        st.write(structured.justification)

        if structured.alternative_queries:
            st.subheader("Alternative Queries")
            for alt in structured.alternative_queries:
                st.write(f"- {alt}")

    except Exception as e:
        st.error(f"Failed to get a structured response from the model: {e}")
