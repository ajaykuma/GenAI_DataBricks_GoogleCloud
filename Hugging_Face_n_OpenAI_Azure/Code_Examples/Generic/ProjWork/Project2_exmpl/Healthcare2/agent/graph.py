# LangGraph agent graph
import os
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
#from langchain_openai import ChatOpenAI
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from .state import AgentState
from .tools import get_patient_history, book_appointment, update_patient_record, search_medical_info
from .memory import retrieve_context
from .prompts import SYSTEM_PROMPT, PLANNER_PROMPT, HELPER_PROMPT

from dotenv import load_dotenv
load_dotenv("E:\\Lesson_2_demos\\.env")

# ── LLM Setup ──────────────
#llm = ChatOpenAI(model="gpt-4.1", temperature=0)
llm = AzureChatOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("API_KEY"),
    api_version="2024-12-01-preview",
    deployment_name="gpt-4.1",
    temperature=0,
)

tools = [get_patient_history, book_appointment, update_patient_record, search_medical_info]
llm_with_tools = llm.bind_tools(tools)

# ── Node 1: Retrieve Context from Memory ───────────
def retrieve_node(state: AgentState) -> AgentState:
    last_message = state["messages"][-1].content
    context = retrieve_context(last_message)
    return {"patient_context": context}

# ── Node 2: Plan Tasks ────────────────────────────────────────────
def planner_node(state: AgentState) -> AgentState:
    last_message = state["messages"][-1].content
    plan_response = llm.invoke(
        PLANNER_PROMPT.format_messages(query=last_message)
    )
    tasks = plan_response.content.strip().split("\n")
    return {"task_plan": tasks}

# ── Node 3: Agent Reasoning + Tool Calls ─────────────────────────
def agent_node(state: AgentState) -> AgentState:
    context = state.get("patient_context", "No prior context available.")
    system_msg = SystemMessage(content=HELPER_PROMPT.format(retrieved_context=context))
    messages = [system_msg] + state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}

# ── Node 4: Tool Execution ──────────────────────────────
tool_node = ToolNode(tools)

# ── Node 5: Summarize Final Response ────────────────
def summarizer_node(state: AgentState) -> AgentState:
    messages = state["messages"]
    # Collect all tool results
    tool_outputs = [m.content for m in messages if hasattr(m, 'role') and m.role == 'tool']
    
    summary_prompt = f"""Based on the conversation and tool results, 
    provide a clear, empathetic summary for the patient/attendant.
    Include: what was found, what was done, and any important next steps.
    
    Tool outputs: {tool_outputs}
    """
    summary = llm.invoke([HumanMessage(content=summary_prompt)])
    return {"final_response": summary.content}

# ── Routing Logic ───────────────────────────────
def should_continue(state: AgentState) -> str:
    last_message = state["messages"][-1]
    # If the LLM made tool calls, route to tool execution
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    # Otherwise, summarize and finish
    return "summarize"

# ── Build Graph ───────────────────────────────
def build_graph():
    graph = StateGraph(AgentState)
    
    # Add nodes
    graph.add_node("retrieve_context", retrieve_node)
    graph.add_node("planner", planner_node)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_node)
    graph.add_node("summarizer", summarizer_node)
    
    # Define edges
    graph.set_entry_point("retrieve_context")
    graph.add_edge("retrieve_context", "planner")
    graph.add_edge("planner", "agent")
    
    # Conditional routing: tools or summarize
    graph.add_conditional_edges(
        "agent",
        should_continue,
        {"tools": "tools", "summarize": "summarizer"}
    )
    # After tools, loop back to agent for next reasoning step
    graph.add_edge("tools", "agent")
    graph.add_edge("summarizer", END)
    
    return graph.compile()

healthcare_agent = build_graph()
