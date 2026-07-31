import os
import openai 
import streamlit as st
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel, Field
from typing_extensions import Literal
from dotenv import load_dotenv
from openai import AzureOpenAI
import matplotlib.pyplot as plt
import networkx as nx


load_dotenv("E:\\Lesson_2_demos\\.env")

client = AzureOpenAI(
    api_key=os.getenv("API_KEY"),
    api_version=os.getenv("AZURE_API_VERSION"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
)

# Step 2: Define state structure to track input, decision, and output
class State(TypedDict):
    input: str  
    decision: str  
    output: str  

class Route(BaseModel):
    step: Literal["fantasy", "sci-fi", "mystery"] = Field(None, description="The next step in the routing process")

# Step 3: Function to determine the story genre using AI
def get_router_response(input_text: str) -> str:
    """Uses AI model to categorize input into a specific genre."""
    response = client.chat.completions.create(model="gpt-4.1",
    messages=[
        {"role": "system", "content": "Route the input to 'fantasy', 'sci-fi', or 'mystery' based on its theme. If unsure, default to 'mystery'."},
        {"role": "user", "content": input_text},
    ])
    genre = response.choices[0].message.content.strip().lower()
    return genre if genre in ["fantasy", "sci-fi", "mystery"] else "mystery"

# Step 4: Define story generation functions for each genre
def generate_fantasy_story(state: State):
    """Creates a fantasy story."""
    response = client.chat.completions.create(model="gpt-4.1",
                    messages=[
                    {"role": "system", "content": "Write a fantasy story based on the input."},
                    {"role": "user", "content": state['input']}],
        max_tokens=500)
    return {"output": response.choices[0].message.content.strip(), "decision": "Fantasy"}

def generate_sci_fi_story(state: State):
    """Creates a sci-fi story."""
    response = client.chat.completions.create(
        model="gpt-4.1",
        messages=[
            {"role": "system", "content": "Write a sci-fi story based on the input."},
            {"role": "user", "content": state['input']}
        ],
        max_tokens=500
    )
    return {"output": response.choices[0].message.content.strip(), "decision": "Sci-Fi"}

def generate_mystery_story(state: State):
    """Creates a mystery story."""
    response = client.chat.completions.create(
        model="gpt-4.1",
        messages=[
            {"role": "system", "content": "Write a mystery story based on the input."},
            {"role": "user", "content": state['input']}
        ],
        max_tokens=500
    )
    return {"output": response.choices[0].message.content.strip(), "decision": "Mystery"}

# Step 5: Routing function to determine which story function to call
def route_request(state: State):
    """Determines the genre and routes accordingly."""
    decision = get_router_response(state["input"])
    return {"decision": decision}

def route_decision(state: State):
    """Maps the decision to the correct function."""
    return {
        "fantasy": "generate_fantasy_story",
        "sci-fi": "generate_sci_fi_story",
        "mystery": "generate_mystery_story",
    }.get(state["decision"], "generate_mystery_story")

# Step 6: Build LangGraph workflow
def build_workflow():
    """Constructs the parallel workflow."""
    workflow = StateGraph(State)
    
    workflow.add_node("generate_fantasy_story", generate_fantasy_story)
    workflow.add_node("generate_sci_fi_story", generate_sci_fi_story)
    workflow.add_node("generate_mystery_story", generate_mystery_story)
    workflow.add_node("route_request", route_request)

    workflow.add_edge(START, "route_request")
    workflow.add_conditional_edges(
        "route_request",
        route_decision,
        {
            "generate_fantasy_story": "generate_fantasy_story",
            "generate_sci_fi_story": "generate_sci_fi_story",
            "generate_mystery_story": "generate_mystery_story",
        },
    )
    workflow.add_edge("generate_fantasy_story", END)
    workflow.add_edge("generate_sci_fi_story", END)
    workflow.add_edge("generate_mystery_story", END)

    return workflow.compile()

# Step 7: Function to visualize the routing workflow
def visualize_workflow():
    """Visualize and save the routing workflow."""

    graph = nx.DiGraph()

    edges = [
        ("START", "route_request"),

        ("route_request", "generate_fantasy_story"),
        ("route_request", "generate_sci_fi_story"),
        ("route_request", "generate_mystery_story"),

        ("generate_fantasy_story", "END"),
        ("generate_sci_fi_story", "END"),
        ("generate_mystery_story", "END")
    ]

    graph.add_edges_from(edges)

    plt.figure(figsize=(11,6))

    pos = {
        "START": (0,2),
        "route_request": (2,2),

        "generate_fantasy_story": (5,4),
        "generate_sci_fi_story": (5,2),
        "generate_mystery_story": (5,0),

        "END": (8,2)
    }

    nx.draw(
        graph,
        pos,
        with_labels=True,
        node_color="lightblue",
        node_size=3200,
        edge_color="gray",
        font_size=9,
        font_weight="bold",
        arrows=True
    )

    plt.title("Conditional Routing Workflow")
    plt.savefig("routing_workflow.png")
    plt.close()

# Step 7: Implement the Streamlit UI
def run_streamlit_app():
    """Creates an interactive UI for story generation."""

    st.title("Genre-Based Story Generator")

    user_input = st.text_input("Enter your story idea", "")

    if st.button("Generate Story"):

        if user_input:

            workflow = build_workflow()

            state = workflow.invoke({"input": user_input})

            st.subheader("Detected Genre:")
            st.write(state["decision"].capitalize())

            st.subheader("Generated Story:")
            st.write(state["output"])

            # Display workflow
            visualize_workflow()

            st.image(
                "routing_workflow.png",
                caption="Conditional Routing LangGraph"
            )

if __name__ == "__main__":
    run_streamlit_app()
