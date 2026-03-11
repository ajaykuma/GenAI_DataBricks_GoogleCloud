# LangGraph agent graph
import os
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

from .state import AgentState
from .tools import (
    get_patient_history,
    search_patient_by_name,
    book_appointment,
    update_patient_record,
    list_all_patients,
    search_medical_info
)
from .memory import retrieve_context
from .prompts import SYSTEM_PROMPT, PLANNER_PROMPT

from dotenv import load_dotenv
load_dotenv()  # ← fixed: removed hardcoded path

# ── LLM Setup ─────────────────────────────────────────────────────
llm = AzureChatOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("API_KEY"),
    api_version=os.getenv("AZURE_API_VERSION"),           # ← from .env
    azure_deployment=os.getenv("AZURE_DEPLOYMENT_NAME"),  # ← from .env
    temperature=0,
)

tools = [
    get_patient_history,
    search_patient_by_name,
    book_appointment,
    update_patient_record,
    list_all_patients,
    search_medical_info
]
llm_with_tools = llm.bind_tools(tools)

# ── Node 1: Retrieve Context from Memory ─────────────────────────
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
    system_msg = SystemMessage(content=SYSTEM_PROMPT.format(retrieved_context=context))
    messages = [system_msg] + state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}

# ── Node 4: Tool Execution ────────────────────────────────────────
tool_node = ToolNode(tools)

# ── Node 5: Summarize Final Response ─────────────────────────────
def summarizer_node(state: AgentState) -> AgentState:
    messages = state["messages"]

    # ← fixed: ToolMessage is the correct type to check, not m.role == 'tool'
    tool_outputs = [
        f"Tool: {m.name}\nResult: {m.content}"
        for m in messages
        if isinstance(m, ToolMessage)
    ]

    tool_summary = "\n\n".join(tool_outputs) if tool_outputs else "No tools were called."

    summary_prompt = f"""Based on the conversation and tool results below,
provide a clear, empathetic summary for the patient or attendant.
Include: what was found, what was done, and any important next steps.
Do not repeat raw JSON — summarize in plain language.

Tool Results:
{tool_summary}
"""
    summary = llm.invoke([HumanMessage(content=summary_prompt)])
    return {"final_response": summary.content}

# ── Routing Logic ─────────────────────────────────────────────────
def should_continue(state: AgentState) -> str:
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return "summarize"

# ── Build Graph ───────────────────────────────────────────────────
def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("retrieve_context", retrieve_node)
    graph.add_node("planner", planner_node)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_node)
    graph.add_node("summarizer", summarizer_node)

    graph.set_entry_point("retrieve_context")
    graph.add_edge("retrieve_context", "planner")
    graph.add_edge("planner", "agent")

    graph.add_conditional_edges(
        "agent",
        should_continue,
        {"tools": "tools", "summarize": "summarizer"}
    )
    graph.add_edge("tools", "agent")
    graph.add_edge("summarizer", END)

    return graph.compile()

healthcare_agent = build_graph()
