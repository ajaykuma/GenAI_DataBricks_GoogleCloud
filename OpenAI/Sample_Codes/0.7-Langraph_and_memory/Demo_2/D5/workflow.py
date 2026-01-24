# ----workflow.py----
from langgraph.graph import StateGraph
import networkx as nx
import matplotlib.pyplot as plt
from agents import intake_agent, evaluation_agent, scheduling_agent, crm_update_agent
from state import InquiryState
from dataclasses import asdict

def construct_graph():
    """Builds the state graph for inquiry processing."""
    workflow = StateGraph(InquiryState)  # Use InquiryState

    # Add agent nodes
    workflow.add_node("intake", intake_agent)
    workflow.add_node("evaluation", evaluation_agent)
    workflow.add_node("scheduling", scheduling_agent)
    workflow.add_node("crm_update", crm_update_agent)

    # Add an "end" node as a dummy function
    workflow.add_node("end", lambda state: asdict(state))  # No processing, just a pass-through

    # Define the workflow structure
    workflow.set_entry_point("intake")  
    workflow.add_edge("intake", "evaluation")
    workflow.add_edge("evaluation", "scheduling")
    workflow.add_edge("scheduling", "crm_update")
    
    # Connect "crm_update" to the "end" node
    workflow.add_edge("crm_update", "end")

    # Mark "end" as the final step
    workflow.set_finish_point("end")  

    return workflow.compile()



def visualize_graph():
    """Generates a visualization of the workflow graph."""
    graph = nx.DiGraph()
    edges = [
        ("START", "intake"),
        ("intake", "evaluation"),
        ("evaluation", "scheduling"),
        ("scheduling", "crm_update")  # Last node (no outgoing edges)
    ]
    graph.add_edges_from(edges)
    
    plt.figure(figsize=(8, 5))
    nx.draw(graph, with_labels=True, node_color='lightblue', edge_color='gray', node_size=2000, font_size=10, font_weight='bold')
    plt.savefig("workflow.png")
    
    return "workflow.png"
