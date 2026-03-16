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
    search_medical_info,
    get_doctor_info,
    add_patient_note
)

from .memory import retrieve_context
from .prompts import SYSTEM_PROMPT, PLANNER_PROMPT

from dotenv import load_dotenv
load_dotenv()

# ── LLM Setup ─────────────────────────────────────────────────────
llm = AzureChatOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("API_KEY"),
    api_version=os.getenv("AZURE_API_VERSION"),
    azure_deployment=os.getenv("AZURE_DEPLOYMENT_NAME"),
    temperature=0,
)

tools = [
    get_patient_history,
    search_patient_by_name,
    book_appointment,
    update_patient_record,
    list_all_patients,
    search_medical_info,
    get_doctor_info,
    add_patient_note
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

    tool_outputs = [
        f"Tool: {m.name}\nResult: {m.content}"
        for m in messages
        if isinstance(m, ToolMessage)
    ]

    # ── No tools called = general health question ─────────────────
    if not tool_outputs:
        # Find last AI response
        ai_messages = [
            m for m in messages
            if hasattr(m, "content")
            and not isinstance(m, ToolMessage)
            and not isinstance(m, HumanMessage)
            and not isinstance(m, SystemMessage)
        ]
        if ai_messages and len(ai_messages[-1].content.strip()) > 20:
            # AI already answered — return directly without re-wrapping
            return {"final_response": ai_messages[-1].content}

        # AI gave no useful answer — ask directly
        original_query = state["messages"][0].content
        direct = llm.invoke([
            SystemMessage(content=(
                "You are a helpful medical assistant. "
                "Answer the question clearly in plain language. "
                "Do NOT frame your answer as a clinical summary or patient report. "
                "Do NOT mention patient records, appointments, or next steps "
                "unless directly asked."
            )),
            HumanMessage(content=original_query)
        ])
        return {"final_response": direct.content}

    # ── Tools were called = patient or search query ───────────────
    tool_summary = "\n\n".join(tool_outputs)
    summary_prompt = f"""Based on the tool results below, provide a clear and 
empathetic summary for the patient or attendant.
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
