#Streamlit UI
import streamlit as st
from langchain_core.messages import HumanMessage
from agent.graph import healthcare_agent
from agent.memory import add_to_memory
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="Healthcare Assistant", layout="wide")
st.title("Agentic Healthcare Assistant")
st.caption("Powered by GPT-4.1 · LangGraph · RAG")

# ── Sidebar ──────────────────────────────────
with st.sidebar:
    st.header("Patient Context")
    patient_id = st.text_input("Patient ID", value="P001")
    st.markdown("---")
    st.header("Add to Memory")
    new_memory = st.text_area("Add patient note:")
    if st.button("Save to Memory") and new_memory:
        add_to_memory(new_memory, {"patient_id": patient_id})
        st.success("Saved to vector memory!")
    st.markdown("---")
    st.header("Example Queries")
    examples = [
        "My 70-year-old father has CKD. Book a nephrologist and summarize latest treatments.",
        "Retrieve history for patient P001",
        "What are the latest guidelines for managing hypertension?",
        "Book a cardiology appointment for John Smith"
    ]
    for ex in examples:
        if st.button(ex[:50] + "...", key=ex):
            st.session_state["prefill"] = ex

# ── Chat Interface ──────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

prefill = st.session_state.pop("prefill", "")
user_input = st.chat_input("Ask the healthcare assistant...") or prefill

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner(" Agent is working..."):
            # Show intermediate steps in expander
            with st.expander(" Agent Reasoning Steps", expanded=False):
                state_placeholder = st.empty()
            
            # Run the agent
            result = healthcare_agent.invoke({
                "messages": [HumanMessage(content=user_input)],
                "patient_id": patient_id
            })
            
            # Display plan
            if result.get("task_plan"):
                with st.expander("Task Plan", expanded=True):
                    for task in result["task_plan"]:
                        if task.strip():
                            st.markdown(f"- {task}")
            
            # Display final response
            final = result.get("final_response", "I processed your request.")
            st.markdown(final)
    
    st.session_state.messages.append({"role": "assistant", "content": final})
