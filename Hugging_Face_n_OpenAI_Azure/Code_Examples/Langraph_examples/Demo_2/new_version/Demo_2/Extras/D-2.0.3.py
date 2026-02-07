import os
import streamlit as st
from typing import List, Dict
from typing_extensions import TypedDict
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import networkx as nx

from langchain_community.chat_models import AzureChatOpenAI
from langchain_core.messages import HumanMessage, BaseMessage, AIMessage
from langgraph.graph import StateGraph, START, END

# -------------------------------
# Load environment
# -------------------------------
load_dotenv("E:\\Lesson_2_demos\\.env")

llm = AzureChatOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("API_KEY"),
    api_version="2024-12-01-preview",
    deployment_name="gpt-4.1",
    temperature=0.7,
)

# -------------------------------
# State definition
# -------------------------------
class State(TypedDict):
    product_name: str
    basic_description: str
    features_benefits: str
    marketing_message_premium: str
    marketing_message_loyalty: str
    marketing_message_corporate: str
    final_description: str
    history: List[BaseMessage]  # store message history
    usage: Dict[str, int]       # prompt_tokens, completion_tokens, total_tokens

# -------------------------------
# Helper functions
# -------------------------------
def accumulate_usage(current: Dict[str, int], new: Dict[str, int]) -> Dict[str, int]:
    return {
        "prompt_tokens": current.get("prompt_tokens", 0) + (new.get("prompt_tokens") or 0),
        "completion_tokens": current.get("completion_tokens", 0) + (new.get("completion_tokens") or 0),
        "total_tokens": current.get("total_tokens", 0) + (new.get("total_tokens") or 0),
    }

def call_llm(state: State, prompt: str, use_memory: bool = False) -> Dict:
    if use_memory:
        ai_message = state["history"][-1] if state.get("history") else AIMessage(content="(No memory)")
        usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        return {"message": ai_message, "usage": usage, "source": "Memory"}

    messages = state.get("history", []) + [HumanMessage(content=prompt)]
    response = llm.generate([messages])
    ai_message = AIMessage(content=response.generations[0][0].text)

    try:
        token_usage = response.llm_output.get("token_usage", {})
        usage = {
            "prompt_tokens": token_usage.get("prompt_tokens", len(prompt.split())),
            "completion_tokens": token_usage.get("completion_tokens", len(ai_message.content.split())),
            "total_tokens": token_usage.get("total_tokens", len(prompt.split()) + len(ai_message.content.split())),
        }
    except Exception:
        usage = {
            "prompt_tokens": len(prompt.split()),
            "completion_tokens": len(ai_message.content.split()),
            "total_tokens": len(prompt.split()) + len(ai_message.content.split()),
        }

    state.setdefault("history", []).append(HumanMessage(content=prompt))
    state["history"].append(ai_message)

    return {"message": ai_message, "usage": usage, "source": "LLM"}

# -------------------------------
# Node functions (unchanged)
# -------------------------------
def generate_basic_description(state: State, use_memory=False):
    if use_memory and state["basic_description"]:
        return {"basic_description": state["basic_description"], "source": "Memory"}
    prompt = f"Write a brief description of a product called '{state['product_name']}'."
    result = call_llm(state, prompt)
    state["basic_description"] = result["message"].content
    state["usage"] = accumulate_usage(state["usage"], result["usage"])
    return {"basic_description": state["basic_description"], "source": result["source"]}

def add_features_benefits(state: State, use_memory=False):
    if use_memory and state["features_benefits"]:
        return {"features_benefits": state["features_benefits"], "source": "Memory"}
    prompt = f"List key features and benefits of this product:\n{state['basic_description']}"
    result = call_llm(state, prompt)
    state["features_benefits"] = result["message"].content
    state["usage"] = accumulate_usage(state["usage"], result["usage"])
    return {"features_benefits": state["features_benefits"], "source": result["source"]}

def create_marketing_message_premium(state: State, use_memory=False):
    if use_memory and state["marketing_message_premium"]:
        return {"marketing_message_premium": state["marketing_message_premium"], "source": "Memory"}
    prompt = f"Create a premium marketing message for the product:\n{state['features_benefits']}"
    result = call_llm(state, prompt)
    state["marketing_message_premium"] = result["message"].content
    state["usage"] = accumulate_usage(state["usage"], result["usage"])
    return {"marketing_message_premium": state["marketing_message_premium"], "source": result["source"]}

def create_marketing_message_loyalty(state: State, use_memory=False):
    if use_memory and state["marketing_message_loyalty"]:
        return {"marketing_message_loyalty": state["marketing_message_loyalty"], "source": "Memory"}
    prompt = f"Create a loyalty-based marketing message for the product:\n{state['features_benefits']}"
    result = call_llm(state, prompt)
    state["marketing_message_loyalty"] = result["message"].content
    state["usage"] = accumulate_usage(state["usage"], result["usage"])
    return {"marketing_message_loyalty": state["marketing_message_loyalty"], "source": result["source"]}

def create_marketing_message_corporate(state: State, use_memory=False):
    if use_memory and state["marketing_message_corporate"]:
        return {"marketing_message_corporate": state["marketing_message_corporate"], "source": "Memory"}
    prompt = f"Create a corporate marketing message for the product:\n{state['features_benefits']}"
    result = call_llm(state, prompt)
    state["marketing_message_corporate"] = result["message"].content
    state["usage"] = accumulate_usage(state["usage"], result["usage"])
    return {"marketing_message_corporate": state["marketing_message_corporate"], "source": result["source"]}

def polish_final_description(state: State, use_memory=False):
    if use_memory and state["final_description"]:
        return {"final_description": state["final_description"], "source": "Memory"}
    prompt = (
        f"Polish the final product description using the following details:\n"
        f"{state['features_benefits']}\n"
        f"Premium: {state['marketing_message_premium']}\n"
        f"Loyalty: {state['marketing_message_loyalty']}\n"
        f"Corporate: {state['marketing_message_corporate']}"
    )
    result = call_llm(state, prompt)
    state["final_description"] = result["message"].content
    state["usage"] = accumulate_usage(state["usage"], result["usage"])
    return {"final_description": state["final_description"], "source": result["source"]}

# -------------------------------
# Workflow builder / visualization
# -------------------------------
def build_workflow():
    workflow = StateGraph(State)
    workflow.add_node("generate_basic_description", generate_basic_description)
    workflow.add_node("add_features_benefits", add_features_benefits)
    workflow.add_node("create_marketing_message_premium", create_marketing_message_premium)
    workflow.add_node("create_marketing_message_loyalty", create_marketing_message_loyalty)
    workflow.add_node("create_marketing_message_corporate", create_marketing_message_corporate)
    workflow.add_node("polish_final_description", polish_final_description)

    workflow.add_edge(START, "generate_basic_description")
    workflow.add_edge("generate_basic_description", "add_features_benefits")
    workflow.add_edge("add_features_benefits", "create_marketing_message_premium")
    workflow.add_edge("add_features_benefits", "create_marketing_message_loyalty")
    workflow.add_edge("add_features_benefits", "create_marketing_message_corporate")
    workflow.add_edge("create_marketing_message_premium", "polish_final_description")
    workflow.add_edge("create_marketing_message_loyalty", "polish_final_description")
    workflow.add_edge("create_marketing_message_corporate", "polish_final_description")
    workflow.add_edge("polish_final_description", END)
    return workflow.compile()

def visualize_workflow():
    graph = nx.DiGraph()
    edges = [
        ("START", "generate_basic_description"),
        ("generate_basic_description", "add_features_benefits"),
        ("add_features_benefits", "create_marketing_message_premium"),
        ("add_features_benefits", "create_marketing_message_loyalty"),
        ("add_features_benefits", "create_marketing_message_corporate"),
        ("create_marketing_message_premium", "polish_final_description"),
        ("create_marketing_message_loyalty", "polish_final_description"),
        ("create_marketing_message_corporate", "polish_final_description"),
        ("polish_final_description", "END"),
    ]
    graph.add_edges_from(edges)
    plt.figure(figsize=(12, 6))
    nx.draw(graph, with_labels=True, node_color='lightblue', edge_color='gray',
            node_size=2000, font_size=10, font_weight='bold')
    plt.savefig("workflow.png")

# -------------------------------
# Session memory and runs
# -------------------------------
PRODUCT_MEMORY: Dict[str, List[State]] = {}  # store all runs for product
MAX_RECENT = 5
RECENT_PRODUCTS: List[str] = []

# -------------------------------
# Run workflow
# -------------------------------
def run_product_workflow(state: State, use_memory: bool):
    """
    Executes the product description workflow.
    - Uses memory if available (skips LLM calls for existing fields)
    - Otherwise, calls the relevant LLM node
    """
    outputs = {}

    nodes = [
        ("Basic Description", generate_basic_description, "basic_description"),
        ("Features & Benefits", add_features_benefits, "features_benefits"),
        ("Premium Marketing", create_marketing_message_premium, "marketing_message_premium"),
        ("Loyalty Marketing", create_marketing_message_loyalty, "marketing_message_loyalty"),
        ("Corporate Marketing", create_marketing_message_corporate, "marketing_message_corporate"),
        ("Final Description", polish_final_description, "final_description")
    ]

    for title, func, key in nodes:
        # If using memory and field already has content, skip LLM call
        if use_memory and state.get(key):
            outputs[title] = {"content": state[key], "source": "Memory"}
        else:
            # Call the LLM function
            result = func(state, use_memory)
            # Ensure the node function updates state[key]
            state[key] = result.get("content", state.get(key, ""))
            outputs[title] = {"content": state[key], "source": result.get("source", "LLM")}

    return outputs

# -------------------------------
# Streamlit app
# -------------------------------
import copy
def run_streamlit_app():
    st.title("Product Description Generator with Memory and Runs")

    # --------------------------------
    # Persistent session memory
    # --------------------------------
    if "PRODUCT_MEMORY" not in st.session_state:
        st.session_state.PRODUCT_MEMORY = {}

    if "RECENT_PRODUCTS" not in st.session_state:
        st.session_state.RECENT_PRODUCTS = []

    product_name = st.text_input("Enter product name:")

    if st.button("Generate") and product_name:
        # Normalize product name (important!)
        product_name = product_name.strip().lower()

        # Session limit tracking
        if product_name not in st.session_state.RECENT_PRODUCTS:
            st.session_state.RECENT_PRODUCTS.append(product_name)

        if len(st.session_state.RECENT_PRODUCTS) > MAX_RECENT:
            st.warning(f"Session limit reached ({MAX_RECENT} products). Refresh page to continue.")
            return

        # --------------------------------
        # Detect memory & run count
        # --------------------------------
        previous_runs = st.session_state.PRODUCT_MEMORY.get(product_name, [])
        run_count = len(previous_runs) + 1
        use_memory = len(previous_runs) > 0

        st.info(f"Generating '{product_name}' (Run {run_count})")

        # --------------------------------
        # Initialize state
        # --------------------------------
        if use_memory:
            # Deep copy last run to avoid mutation
            state = copy.deepcopy(previous_runs[-1])
        else:
            state = State(
                product_name=product_name,
                history=[],
                basic_description="",
                features_benefits="",
                marketing_message_premium="",
                marketing_message_loyalty="",
                marketing_message_corporate="",
                final_description="",
                usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
            )

        # --------------------------------
        # Run workflow
        # --------------------------------
        outputs = run_product_workflow(state, use_memory)

        # --------------------------------
        # Persist run to memory
        # --------------------------------
        st.session_state.PRODUCT_MEMORY.setdefault(product_name, []).append(
            copy.deepcopy(state)
        )

        # REQUIRED DEBUG LINE
        st.write(
            "MEMORY SNAPSHOT:",
            {k: len(v) for k, v in st.session_state.PRODUCT_MEMORY.items()}
        )

        # --------------------------------
        # Display outputs
        # --------------------------------
        TITLE_KEY_MAP = {
            "Basic Description": "basic_description",
            "Features & Benefits": "features_benefits",
            "Premium Marketing": "marketing_message_premium",
            "Loyalty Marketing": "marketing_message_loyalty",
            "Corporate Marketing": "marketing_message_corporate",
            "Final Description": "final_description"
        }

        st.subheader(f"All Runs for '{product_name}'")

        for idx, past_state in enumerate(st.session_state.PRODUCT_MEMORY[product_name], 1):
            st.markdown(f"### Run {idx}")

            for title, key in TITLE_KEY_MAP.items():
                if past_state.get(key):
                    source = "Memory" if idx < run_count else outputs[title]["source"]
                    st.markdown(f"**{title} ({source})**")
                    st.write(past_state[key])

        # --------------------------------
        # Token usage (last run)
        # --------------------------------
        st.subheader("Token Usage Summary (Last Run)")
        usage = state["usage"]
        st.write(f"Prompt tokens: {usage['prompt_tokens']}")
        st.write(f"Completion tokens: {usage['completion_tokens']}")
        st.write(f"Total tokens: {usage['total_tokens']}")

        # --------------------------------
        # Workflow visualization
        # --------------------------------
        visualize_workflow()
        st.image("workflow.png", caption="Workflow Graph")

if __name__ == "__main__":
    run_streamlit_app()
