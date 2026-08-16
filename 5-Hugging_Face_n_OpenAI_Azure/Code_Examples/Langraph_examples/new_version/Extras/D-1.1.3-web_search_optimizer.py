''' 
#The google-search-results package (the official SerpAPI client)
pip install google-search-results

Split into three stages instead of one LLM call:
Plan — LLM optimizes the query (no longer guessed from memory)
Search — run_serpapi_search() actually hits SerpAPI with that query and pulls back 
         title/snippet/link for the top results
Answer — a second LLM call answers the user's question using only those search results, 
         with an explicit instruction not to use outside knowledge
insufficient_evidence flag — if the search results don't actually contain an answer, 
                             the model has to say so (st.warning) instead of quietly filling the 
                             gap with a guess. 
                             This is the main payoff of grounding: you can now tell the difference 
                             between "answered from real sources" and "couldn't find it."
Sources are shown and traceable — each result is listed with a marker if the model actually cited 
it in used_source_numbers, so you can sanity-check whether the answer is really coming from the 
sources or the model padded it.
@st.cache_data(ttl=3600) on the SerpAPI call — reruns for the same query within an hour won't 
burn API quota re-fetching identical results.

A couple of things worth deciding on:

Number of results — currently pulls 5. More gives the model richer grounding but costs 
more tokens/latency; fewer is faster but riskier if the top result is thin.
SerpAPI quota — free tier is 100 searches/month, so if you're testing heavily you'll want to watch 
for RuntimeError: SerpAPI error from rate limiting.

'''
import os
from typing import List, Optional

import streamlit as st
from pydantic import BaseModel, Field
from langchain_openai.chat_models import AzureChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv
from serpapi import GoogleSearch

load_dotenv()

client = AzureChatOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("API_KEY"),
    api_version="2025-01-01-preview",
    deployment_name="gpt-4.1",
    temperature=0,
)

SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")


# --------------------------------------------------------------------
# Stage 1 schema: query optimization only. No "answer" here anymore --
# the answer now comes from real search results in Stage 3, not the
# model's memory.
# --------------------------------------------------------------------
class SearchQueryPlan(BaseModel):
    search_query: str = Field(description="A concise, optimized web search query")
    justification: str = Field(description="Why this search query was chosen, in 1-2 sentences")
    alternative_queries: List[str] = Field(
        default_factory=list,
        description="2-3 alternative search queries targeting different angles",
    )


# --------------------------------------------------------------------
# Stage 3 schema: the grounded answer, built only from the search
# results we actually fetched.
# --------------------------------------------------------------------
class GroundedAnswer(BaseModel):
    answer: str = Field(description="A direct answer to the user's question, based only on the provided search results")
    used_source_numbers: List[int] = Field(
        default_factory=list,
        description="Which numbered sources (1-indexed) the answer actually relied on",
    )
    insufficient_evidence: bool = Field(
        default=False,
        description="True if the search results did not contain enough information to answer confidently",
    )


planner = client.with_structured_output(SearchQueryPlan)
answerer = client.with_structured_output(GroundedAnswer)

PLANNER_SYSTEM_PROMPT = """You turn a user's question into an optimized web search query.

Rules:
- justification: Explain what makes the search_query effective (specific terminology, \
disambiguating words, scoping to docs/tutorials/github). Do not restate or paraphrase \
the user's question -- that's not a justification.
- alternative_queries: Each one should target a genuinely different angle (official docs, \
a tutorial, GitHub issues/discussions, a comparison), not a reworded version of the same query.
- search_query: Keep it to what you'd actually type into a search engine -- short, \
specific, no filler words.
"""

ANSWERER_SYSTEM_PROMPT = """You answer the user's question using ONLY the numbered search \
results provided below. Do not use outside knowledge.

Rules:
- If the results answer the question, give a direct, specific answer and list which \
source numbers you relied on.
- If the results are insufficient, irrelevant, or contradictory, set \
insufficient_evidence=true and say so plainly in `answer` rather than guessing.
- Do not fabricate URLs, numbers, or claims that aren't supported by the results below.
"""


@st.cache_data(show_spinner=False, ttl=3600)
def run_serpapi_search(query: str, num_results: int = 5) -> List[dict]:
    """Fetch organic results from SerpAPI. Cached per-query for an hour so
    repeated identical questions don't burn API quota."""
    if not SERPAPI_API_KEY:
        raise RuntimeError("SERPAPI_API_KEY not found in environment (.env)")

    search = GoogleSearch({
        "q": query,
        "api_key": SERPAPI_API_KEY,
        "num": num_results,
    })
    data = search.get_dict()

    if "error" in data:
        raise RuntimeError(f"SerpAPI error: {data['error']}")

    results = []
    for item in data.get("organic_results", [])[:num_results]:
        results.append({
            "title": item.get("title", ""),
            "link": item.get("link", ""),
            "snippet": item.get("snippet", ""),
        })
    return results


def format_results_for_prompt(results: List[dict]) -> str:
    lines = []
    for i, r in enumerate(results, start=1):
        lines.append(f"[{i}] {r['title']}\n{r['snippet']}\nURL: {r['link']}")
    return "\n\n".join(lines)


st.title("Web Search Optimization with LLM")
user_query = st.text_input("Enter your question:")

if user_query:
    # ---------------- Stage 1: optimize the query ----------------
    try:
        plan: SearchQueryPlan = planner.invoke([
            SystemMessage(content=PLANNER_SYSTEM_PROMPT),
            HumanMessage(content=f'User question: "{user_query}"'),
        ])
    except Exception as e:
        st.error(f"Failed to generate an optimized search query: {e}")
        st.stop()

    st.subheader("Optimized Search Query")
    st.write(plan.search_query)

    st.subheader("Reasoning")
    st.write(plan.justification)

    if plan.alternative_queries:
        st.subheader("Alternative Queries")
        for alt in plan.alternative_queries:
            st.write(f"- {alt}")

    # ---------------- Stage 2: run the real search ----------------
    st.divider()
    try:
        with st.spinner("Searching..."):
            results = run_serpapi_search(plan.search_query)
    except RuntimeError as e:
        st.error(str(e))
        st.stop()

    if not results:
        st.warning("No search results found for this query.")
        st.stop()

    # ---------------- Stage 3: ground the answer in real results ----------------
    try:
        grounded: GroundedAnswer = answerer.invoke([
            SystemMessage(content=ANSWERER_SYSTEM_PROMPT),
            HumanMessage(
                content=(
                    f'User question: "{user_query}"\n\n'
                    f"Search results:\n{format_results_for_prompt(results)}"
                )
            ),
        ])
    except Exception as e:
        st.error(f"Failed to generate a grounded answer: {e}")
        st.stop()

    st.subheader("Answer")
    if grounded.insufficient_evidence:
        st.warning(grounded.answer)
    else:
        st.write(grounded.answer)

    st.subheader("Sources")
    for i, r in enumerate(results, start=1):
        used = i in grounded.used_source_numbers
        marker = " used" if used else "—"
        st.markdown(f"**[{i}] {marker}** [{r['title']}]({r['link']})")
        st.caption(r["snippet"])
