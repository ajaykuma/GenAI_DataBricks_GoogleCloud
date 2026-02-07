import os
import streamlit as st
from typing_extensions import TypedDict
from typing import List, Union, Dict, Annotated
#from difflib import SequenceMatcher
from dotenv import load_dotenv
import langchain
import langchain_community
import langchain_core
from langgraph.graph import StateGraph, START, END
from langchain_community.chat_models import AzureChatOpenAI
#from langchain_community.memory import ConversationBufferWindowMemory
#from langchain_community.chains import ConversationChain
from langchain_core.messages import HumanMessage, BaseMessage, AIMessage

import matplotlib.pyplot as plt
import networkx as nx

# print(langchain.__version__)
# print(langchain_community.__version__)
# print(langchain_core.__version__)

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

#State definition (Memory that will be used)

class State(TypedDict):
    product_name: str
    basic_description: str
    features_benefits: str
    marketing_message_premium: str
    marketing_message_loyalty: str
    marketing_message_corporate: str
    final_description: str
    history: Annotated[List[BaseMessage], "append"]  
    usage: Dict[str, int]  # prompt_tokens, completion_tokens, total_tokens

# -------------------------------
# Helper function to safely add usage
# -------------------------------
def accumulate_usage(current: Dict[str, int], new: Dict[str, int]) -> Dict[str, int]:
    return {
        "prompt_tokens": current.get("prompt_tokens", 0) + (new.get("prompt_tokens") or 0),
        "completion_tokens": current.get("completion_tokens", 0) + (new.get("completion_tokens") or 0),
        "total_tokens": current.get("total_tokens", 0) + (new.get("total_tokens") or 0),
    }

def call_llm(state: dict, prompt: str) -> dict:
    """
    Call AzureChatOpenAI using langchain-community 0.4.1 style.
    """

    messages = state.get("history", []) + [HumanMessage(content=prompt)]

    # Use .generate() method
    response = llm.generate([messages])  # note the double list: batch of message sequences

    # Extract content from response
    ai_message = AIMessage(content=response.generations[0][0].text)

    # Append user + AI message to history
    state.setdefault("history", []).append(HumanMessage(content=prompt))
    state["history"].append(ai_message)

    # Return structure similar to before
    usage = {
        "prompt_tokens": None,
        "completion_tokens": None,
        "total_tokens": None,
    }

    return {
        "message": ai_message,
        "usage": usage
    }

# -------------------------------
# Node implementations
# -------------------------------
def generate_basic_description(state: State):
    prompt = f"Write a brief description of a product called '{state['product_name']}'."
    result = call_llm(state, prompt)

    # Append safely to annotated list
    state["history"].append(HumanMessage(content=prompt))
    state["history"].append(result["message"])
    state["usage"] = accumulate_usage(state["usage"], result["usage"])

    return {
        "basic_description": result["message"].content
    }


def add_features_benefits(state: State):
    prompt = f"List key features and benefits of this product:\n{state['basic_description']}"
    result = call_llm(state, prompt)

    state["history"].append(HumanMessage(content=prompt))
    state["history"].append(result["message"])
    state["usage"] = accumulate_usage(state["usage"], result["usage"])

    return {
        "features_benefits": result["message"].content
    }


def create_marketing_message_premium(state: State):
    prompt = f"Create a premium marketing message for the product:\n{state['features_benefits']}"
    result = call_llm(state, prompt)

    state["history"].append(HumanMessage(content=prompt))
    state["history"].append(result["message"])
    state["usage"] = accumulate_usage(state["usage"], result["usage"])

    return {
        "marketing_message_premium": result["message"].content
    }


def create_marketing_message_loyalty(state: State):
    prompt = f"Create a loyalty-based marketing message for the product:\n{state['features_benefits']}"
    result = call_llm(state, prompt)

    state["history"].append(HumanMessage(content=prompt))
    state["history"].append(result["message"])
    state["usage"] = accumulate_usage(state["usage"], result["usage"])

    return {
        "marketing_message_loyalty": result["message"].content
    }


def create_marketing_message_corporate(state: State):
    prompt = f"Create a corporate marketing message for the product:\n{state['features_benefits']}"
    result = call_llm(state, prompt)

    state["history"].append(HumanMessage(content=prompt))
    state["history"].append(result["message"])
    state["usage"] = accumulate_usage(state["usage"], result["usage"])

    return {
        "marketing_message_corporate": result["message"].content
    }


def polish_final_description(state: State):
    prompt = (
        f"Polish the final product description using the following details:\n"
        f"{state['features_benefits']}\n"
        f"Premium: {state['marketing_message_premium']}\n"
        f"Loyalty: {state['marketing_message_loyalty']}\n"
        f"Corporate: {state['marketing_message_corporate']}"
    )
    result = call_llm(state, prompt)

    state["history"].append(HumanMessage(content=prompt))
    state["history"].append(result["message"])
    state["usage"] = accumulate_usage(state["usage"], result["usage"])

    return {
        "final_description": result["message"].content
    }
# -------------------------------
# Build LangGraph workflow
# -------------------------------
def build_workflow():
    workflow = StateGraph(State)

    workflow.add_node("generate_basic_description", generate_basic_description)
    workflow.add_node("add_features_benefits", add_features_benefits)
    workflow.add_node("create_marketing_message_premium", create_marketing_message_premium)
    workflow.add_node("create_marketing_message_loyalty", create_marketing_message_loyalty)
    workflow.add_node("create_marketing_message_corporate", create_marketing_message_corporate)
    workflow.add_node("polish_final_description", polish_final_description)

    # edges
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

# Optional: visualize workflow
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
    plt.figure(figsize=(10, 6))
    nx.draw(graph, with_labels=True, node_color='lightblue', edge_color='gray', node_size=2000, font_size=10, font_weight='bold')
    plt.savefig("workflow.png")

# Session counters
PRODUCT_COUNTER = {}
RECENT_PRODUCTS = []
MAX_RECENT = 5

# -------------------------------
# Streamlit app
# -------------------------------
def run_streamlit_app():
    st.title("Product Description Generator (LangGraph + Memory tracking)")

    product_name = st.text_input("Enter product name:")

    if st.button("Generate") and product_name:
        # Handle recent products limit
        if product_name not in RECENT_PRODUCTS:
            RECENT_PRODUCTS.append(product_name)
        if len(RECENT_PRODUCTS) > MAX_RECENT:
            st.warning(f"Session limit reached ({MAX_RECENT} products). Refresh page to continue.")
            return

        # Increment counter
        PRODUCT_COUNTER[product_name] = PRODUCT_COUNTER.get(product_name, 0) + 1
        first_time = PRODUCT_COUNTER[product_name] == 1

        st.info(f"{'First time generating' if first_time else 'Memory reused'} '{product_name}'")
        st.info(f"'{product_name}' requested {PRODUCT_COUNTER[product_name]} time(s) this session")

        # Initial state
        initial_state = State(
            product_name=product_name,
            history=[],  # <-- memory/history stored here
            basic_description="",
            features_benefits="",
            marketing_message_premium="",
            marketing_message_loyalty="",
            marketing_message_corporate="",
            final_description="",
            usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        )

        # Run workflow
        chain = build_workflow()
        result = chain.invoke(initial_state)

        # Display all node outputs
        st.subheader("Node Outputs")
        for node in ["generate_basic_description", "add_features_benefits",
                     "create_marketing_message_premium", "create_marketing_message_loyalty",
                     "create_marketing_message_corporate", "polish_final_description"]:
            if node in result:
                st.markdown(f"**{node.replace('_',' ').title()}:**")
                st.write(result[node])

        # -------------------------------
        # Display memory / history
        # -------------------------------
        st.subheader("Memory / History")
        if "history" in result and result["history"]:
            for i, msg in enumerate(result["history"]):
                role = getattr(msg, "type", "HUMAN").upper()
                st.markdown(f"**{role}** [{i+1}]: {msg.content[:200]}{'...' if len(msg.content)>200 else ''}")
        else:
            st.write("[No history recorded]")

        # -------------------------------
        # Display usage statistics
        # -------------------------------
        st.subheader("Token Usage Summary")
        usage = result.get("usage", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})
        st.write(f"Prompt tokens: {usage.get('prompt_tokens', 0)}")
        st.write(f"Completion tokens: {usage.get('completion_tokens', 0)}")
        st.write(f"Total tokens: {usage.get('total_tokens', 0)}")

        # Optional: visualize workflow
        visualize_workflow()
        st.image("workflow.png", caption="Workflow")

if __name__ == "__main__":
    run_streamlit_app()
