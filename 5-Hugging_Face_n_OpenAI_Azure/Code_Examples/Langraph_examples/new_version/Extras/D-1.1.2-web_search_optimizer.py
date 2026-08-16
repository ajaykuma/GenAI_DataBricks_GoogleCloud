'''With better prompt-quality
So the prompt-quality fix is better than previous. 
The one caveat still stands from before: this answer is the model's best guess from training data, 
not verified against anything current. 
For LangGraph specifically — a fast-moving library — that's a real risk; APIs and testing conventions 
can shift between versions, and the model won't know if its answer is stale.
'''
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
# Schema now matches everything the prompt actually asks for, so the
# model has a labeled slot for each piece 

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

SYSTEM_PROMPT = """You turn a user's question into an optimized web search query, \
and you separately answer the question yourself from your own knowledge.

Rules:
- one_line_answer: Give a real, substantive answer using what you actually know. \
Never say things like "check the documentation," "refer to the official site," \
or "follow the setup guide" -- that is a non-answer. If you know the answer, state it.
- justification: Explain what makes the search_query effective (e.g. specific \
terminology, disambiguating words, scoping to docs/tutorials/github). Do not \
restate or paraphrase the user's question -- that's not a justification.
- alternative_queries: Each one should target a genuinely different angle \
(e.g. official docs, a tutorial, GitHub issues/discussions, a comparison), \
not just a reworded version of the same query.
- search_query: Keep it to what you'd actually type into a search engine -- \
short, specific, no filler words.
"""

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
