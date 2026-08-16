import os
import json
import requests
import matplotlib.pyplot as plt
import networkx as nx
import streamlit as st

from typing_extensions import TypedDict
from openai import AzureOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command
from dotenv import load_dotenv


# ============================================================
# 1. ENVIRONMENT
# ============================================================

load_dotenv("E:\\Lesson_2_demos\\.env")

client = AzureOpenAI(
    api_key=os.getenv("API_KEY"),
    api_version=os.getenv("AZURE_API_VERSION"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
)

SERPAPI_KEY = os.getenv("SERPAPI_KEY")


# ============================================================
# 2. STATE
# ============================================================

class State(TypedDict, total=False):

    # User request
    user_request: str
    product_name: str

    # Agent planning
    plan: str

    # New
    information_source: str
    # Web search
    search_query: str
    search_results: str

    next_action: str

    # Product generation
    basic_description: str
    features_benefits: str
    marketing_message: str
    final_description: str

  
    # Agent evaluation
    evaluation: str
    score: float

    # Agent control
    iteration: int
    max_iterations: int
    goal_achieved: bool


# ============================================================
# 3. LLM HELPER
# ============================================================

def call_llm(system_prompt, user_prompt):

    response = client.chat.completions.create(
        model="gpt-4.1",
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ],
        temperature=0.2
    )

    return response.choices[0].message.content

def decide_information_source(state: State):

    decision = call_llm(
        """
        You are an information-routing agent.

        Determine whether the user's request can be answered
        using the LLM's existing knowledge or whether fresh
        external information is required.

        Choose exactly one:

        LLM
        SERPAPI

        Use SERPAPI when the request involves:
        - latest information
        - current trends
        - current competitors
        - recent prices
        - current events
        - web research
        - information that may have changed recently

        Use LLM when:
        - the task is creative writing
        - rewriting
        - summarization
        - brainstorming
        - generic product description
        - reasoning that does not require current information

        Return only:
        LLM
        or
        SERPAPI
        """,
        f"""
        User request:
        {state["user_request"]}

        Product:
        {state["product_name"]}
        """
    ).strip().upper()

    if decision not in ["LLM", "SERPAPI"]:
        decision = "LLM"

    return {
        "information_source": decision
    }

# ============================================================
# 4. SERPAPI TOOL
# ============================================================

SERPAPI_KEY = os.getenv("SERPAPI_API_KEY")

def serpapi_search(query):

    if not SERPAPI_KEY:
        return "SERPAPI_KEY is not configured."

    url = "https://serpapi.com/search.json"

    params = {
        "engine": "google",
        "q": query,
        "api_key": SERPAPI_KEY,
        "num": 5
    }

    response = requests.get(url, params=params, timeout=20)

    if response.status_code != 200:
        return f"Search failed: {response.text}"

    data = response.json()

    results = []

    for result in data.get("organic_results", [])[:5]:

        results.append({
            "title": result.get("title"),
            "snippet": result.get("snippet"),
            "link": result.get("link")
        })

    return json.dumps(results, indent=2)


# ============================================================
# 5. INITIAL PLANNER
# ============================================================

def planner(state: State):

    prompt = f"""
You are the planning component of an autonomous product marketing agent.

User request:
{state["user_request"]}

Product:
{state["product_name"]}

Create a short execution plan.

The agent has the following capabilities:

1. Generate product description
2. Generate features and benefits
3. Search the web using SerpAPI
4. Generate marketing message
5. Evaluate the result
6. Revise the result

Determine which actions are useful for achieving the user's goal.

Do not assume that every action must be executed.

Return a concise numbered plan.
"""

    plan = call_llm(
        "You are an autonomous planning agent.",
        prompt
    )

    return {
        "plan": plan,
        "iteration": 0,
        "max_iterations": 3
    }


# ============================================================
# 6. DECISION MAKER / ROUTER
# ============================================================

def decision_maker(state: State):

    if state.get("iteration", 0) >= state.get("max_iterations", 3):
        return {
            "next_action": "FINISH"
        }

    prompt = f"""
You are the decision-making component of an autonomous agent.

User request:
{state["user_request"]}

Product:
{state["product_name"]}

Current plan:
{state.get("plan", "")}

Current state:

Basic description:
{state.get("basic_description", "")}

Features and benefits:
{state.get("features_benefits", "")}

Marketing message:
{state.get("marketing_message", "")}

Final description:
{state.get("final_description", "")}

Search results:
{state.get("search_results", "")}

Evaluation:
{state.get("evaluation", "")}

Current iteration:
{state.get("iteration", 0)}

Available actions:

GENERATE_BASIC
Generate a basic product description.

GENERATE_FEATURES
Generate product features and benefits.

WEB_SEARCH
Search the web for useful product/market/competitor information.

GENERATE_MARKETING
Generate a marketing message.

EVALUATE
Evaluate the current output.

REVISE
Improve the current output based on evaluation.

FINISH
Finish because the user's goal has been achieved.

Decide the SINGLE best next action.

Return ONLY one of:

GENERATE_BASIC
GENERATE_FEATURES
WEB_SEARCH
GENERATE_MARKETING
EVALUATE
REVISE
FINISH
"""

    action = call_llm(
        "You are a decision-making controller for an autonomous agent.",
        prompt
    ).strip().upper()

    valid_actions = {
        "GENERATE_BASIC",
        "GENERATE_FEATURES",
        "WEB_SEARCH",
        "GENERATE_MARKETING",
        "EVALUATE",
        "REVISE",
        "FINISH"
    }

    if action not in valid_actions:
        action = "EVALUATE"

    return {
        "next_action": action
    }

def route_information_source(state):

    if state["information_source"] == "SERPAPI":
        return "web_search"

    return "llm_generation"

# ============================================================
# 7. GENERATE BASIC DESCRIPTION
# ============================================================

def generate_basic_description(state: State):

    description = call_llm(
        "You generate concise and accurate product descriptions.",
        f"""
Create a basic product description.

Product:
{state["product_name"]}

User request:
{state["user_request"]}
"""
    )

    return {
        "basic_description": description
    }


# ============================================================
# 8. GENERATE FEATURES
# ============================================================

def generate_features(state: State):

    features = call_llm(
        "You identify useful product features and customer benefits.",
        f"""
Product:
{state["product_name"]}

Basic description:
{state.get("basic_description", "")}

Web research:
{state.get("search_results", "")}

Identify important product features and customer benefits.
Do not invent specific factual claims unless supported by the
available information.
"""
    )

    return {
        "features_benefits": features
    }


# ============================================================
# 9. WEB SEARCH TOOL NODE
# ============================================================

def web_search(state: State):

    query = call_llm(
        "You generate concise search queries for web research.",
        f"""
User request:
{state["user_request"]}

Product:
{state["product_name"]}

Generate ONE useful Google search query.

The purpose is to gather information that would improve the
product marketing output.

Return only the search query.
"""
    ).strip()

    results = serpapi_search(query)

    return {
        "search_query": query,
        "search_results": results
    }


# ============================================================
# 10. GENERATE MARKETING MESSAGE
# ============================================================

def generate_marketing(state: State):

    marketing = call_llm(
        "You are an expert product marketing specialist.",
        f"""
Create a compelling marketing message.

Product:
{state["product_name"]}

Basic description:
{state.get("basic_description", "")}

Features and benefits:
{state.get("features_benefits", "")}

Web research:
{state.get("search_results", "")}

User request:
{state["user_request"]}

Use the available information but avoid unsupported factual claims.
"""
    )

    return {
        "marketing_message": marketing,
        "final_description": marketing
    }


# ============================================================
# 11. EVALUATOR
# ============================================================

def evaluate(state: State):

    evaluation = call_llm(
        "You are a strict quality evaluator for an autonomous marketing agent.",
        f"""
Evaluate the current product marketing output.

User request:
{state["user_request"]}

Product:
{state["product_name"]}

Output:
{state.get("final_description", "")}

Evaluate:

1. Relevance
2. Completeness
3. Accuracy
4. Marketing quality
5. Clarity
6. Alignment with the user's request

Give a score from 0 to 10.

Return exactly this format:

SCORE: <number>

FEEDBACK:
<short explanation>
"""
    )

    score = 0

    try:
        score_line = [
            line for line in evaluation.splitlines()
            if line.startswith("SCORE:")
        ]

        if score_line:
            score = float(
                score_line[0].replace("SCORE:", "").strip()
            )

    except Exception:
        score = 0

    goal_achieved = score >= 8

    return {
        "evaluation": evaluation,
        "score": score,
        "goal_achieved": goal_achieved
    }



# ============================================================
# 12. REVISION
# ============================================================

def revise(state: State):

    revised = call_llm(
        "You revise marketing content based on evaluator feedback.",
        f"""
Improve the following product marketing output.

Product:
{state["product_name"]}

Current output:
{state.get("final_description", "")}

Evaluator feedback:
{state.get("evaluation", "")}

Web research:
{state.get("search_results", "")}

User request:
{state["user_request"]}

Produce an improved version.
"""
    )

    return {
        "final_description": revised,
        "iteration": state.get("iteration", 0) + 1
    }


# ============================================================
# 13. ROUTING FUNCTION
# ============================================================

def route_action(state: State):

    action = state.get("next_action")

    if action == "GENERATE_BASIC":
        return "generate_basic"

    if action == "GENERATE_FEATURES":
        return "generate_features"

    if action == "WEB_SEARCH":
        return "web_search"

    if action == "GENERATE_MARKETING":
        return "generate_marketing"

    if action == "EVALUATE":
        return "evaluate"

    if action == "REVISE":
        return "revise"

    if action == "FINISH":
        return "finish"

    return "evaluate"


# ============================================================
# 14. BUILD AGENTIC LANGGRAPH
# ============================================================

def build_workflow():

    workflow = StateGraph(State)

    # Nodes
    workflow.add_node("planner", planner)
    workflow.add_node("decide_information_source", decide_information_source)
    workflow.add_node("decision_maker", decision_maker)

    workflow.add_node("generate_basic", generate_basic_description)
    workflow.add_node("generate_features", generate_features)
    workflow.add_node("web_search", web_search)
    workflow.add_node("generate_marketing", generate_marketing)
    workflow.add_node("evaluate", evaluate)
    workflow.add_node("revise", revise)

    # Start
    workflow.add_edge(
    START,
    "planner")

    workflow.add_edge(
                "planner",
                "decide_information_source"
            )

    workflow.add_conditional_edges(
                "decide_information_source",
                route_information_source,
                {
                    "web_search": "web_search",
                    "llm_generation": "decision_maker"
                }
            )

                # Decision maker dynamically selects action
    workflow.add_conditional_edges(
                    "decision_maker",

        route_action,
        {
            "generate_basic": "generate_basic",
            "generate_features": "generate_features",
            "web_search": "web_search",
            "generate_marketing": "generate_marketing",
            "evaluate": "evaluate",
            "revise": "revise",
            "finish": END
        }
    )

    # Every action returns control to the decision maker
    workflow.add_edge(
        "generate_basic",
        "decision_maker"
    )

    workflow.add_edge(
        "generate_features",
        "decision_maker"
    )

    workflow.add_edge(
        "web_search",
        "decision_maker"
    )

    workflow.add_edge(
        "generate_marketing",
        "decision_maker"
    )

    workflow.add_edge(
        "evaluate",
        "decision_maker"
    )

    workflow.add_edge(
        "revise",
        "decision_maker"
    )

    return workflow.compile()


# ============================================================
# 15. VISUALIZE GRAPH
# ============================================================

def visualize_workflow():

    graph = nx.DiGraph()

    edges = [
        ("START", "PLANNER"),

        ("PLANNER", "INFORMATION ROUTER"),

        ("INFORMATION ROUTER", "WEB SEARCH"),
        ("INFORMATION ROUTER", "DECISION MAKER"),

        ("WEB SEARCH", "DECISION MAKER"),

        ("DECISION MAKER", "GENERATE BASIC"),
        ("DECISION MAKER", "GENERATE FEATURES"),
        ("DECISION MAKER", "WEB SEARCH"),
        ("DECISION MAKER", "GENERATE MARKETING"),
        ("DECISION MAKER", "EVALUATE"),
        ("DECISION MAKER", "REVISE"),
        ("DECISION MAKER", "END"),

        ("GENERATE BASIC", "DECISION MAKER"),
        ("GENERATE FEATURES", "DECISION MAKER"),
        ("GENERATE MARKETING", "DECISION MAKER"),
        ("EVALUATE", "DECISION MAKER"),
        ("REVISE", "DECISION MAKER")
    ]

    graph.add_edges_from(edges)

    plt.figure(figsize=(14, 8))

    nx.draw(
        graph,
        with_labels=True,
        node_size=2500,
        font_size=8,
        font_weight="bold",
        arrows=True
    )

    plt.savefig(
        "agentic_workflow.png",
        bbox_inches="tight"
    )

    plt.close()


# ============================================================
# 16. STREAMLIT APPLICATION
# ============================================================

def run_streamlit_app():

    st.title(
        "Agentic Product Description Generator"
    )

    product_name = st.text_input(
        "Enter product name:"
    )

    user_request = st.text_area(
        "What would you like the agent to do?",
        value=(
            "Create a compelling marketing description "
            "for this product. Research competitors if "
            "useful."
        )
    )

    if st.button("Run Agent"):

        if not product_name:
            st.warning("Please enter a product name.")
            return

        initial_state = {
            "product_name": product_name,
            "user_request": user_request,
            "iteration": 0,
            "max_iterations": 3
        }

        chain = build_workflow()

        with st.spinner("Agent is working..."):

            result = chain.invoke(initial_state)

        # ----------------------------------------------------
        # RESULTS
        # ----------------------------------------------------

        st.subheader("Agent Plan")

        st.write(
            result.get("plan", "")
        )

        st.subheader("Final Description")

        st.write(
            result.get(
                "final_description",
                ""
            )
        )

        st.subheader("Evaluation")

        st.write(
            result.get(
                "evaluation",
                "Not evaluated"
            )
        )

        st.metric(
            "Quality Score",
            result.get("score", 0)
        )

        st.subheader("Agent Decision")

        st.write(
            result.get(
                "next_action",
                ""
            )
        )

        st.subheader("Iterations")

        st.write(
            result.get(
                "iteration",
                0
            )
        )

        # ----------------------------------------------------
        # INTERMEDIATE INFORMATION
        # ----------------------------------------------------

        with st.expander("Basic Description"):

            st.write(
                result.get(
                    "basic_description",
                    ""
                )
            )

        with st.expander("Features & Benefits"):

            st.write(
                result.get(
                    "features_benefits",
                    ""
                )
            )

        with st.expander("Web Search"):

            st.write(
                "Query:"
            )

            st.write(
                result.get(
                    "search_query",
                    ""
                )
            )

            st.write(
                result.get(
                    "search_results",
                    ""
                )
            )

        # ----------------------------------------------------
        # GRAPH
        # ----------------------------------------------------

        visualize_workflow()

        st.image(
            "agentic_workflow.png",
            caption="Agentic LangGraph Workflow"
        )


# ============================================================
# 17. MAIN
# ============================================================

if __name__ == "__main__":
    run_streamlit_app()
