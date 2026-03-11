# Streamlit UI
import streamlit as st
from langchain_core.messages import HumanMessage
from agent.graph import healthcare_agent
from agent.memory import add_to_memory, ingest_single_pdf
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="Healthcare Assistant", layout="wide")
st.title("Agentic Healthcare Assistant")
st.caption("Powered by GPT-4.1 · LangGraph · RAG")

# ── Single Sidebar Block ──────────────────────────────────────────
with st.sidebar:

    # -- Patient Context --
    st.header("Patient Context")
    patient_id = st.text_input("Patient ID", value="P001")

    st.markdown("---")

    # -- Add to Memory --
    st.header(" Add to Memory")
    new_memory = st.text_area("Add patient note:")
    if st.button("Save to Memory") and new_memory:
        add_to_memory(new_memory, {"patient_id": patient_id})
        st.success("Saved to vector memory!")

    st.markdown("---")

    # -- Upload PDF --
    st.header(" Upload Patient PDF")
    uploaded_pdf = st.file_uploader("Upload a report or record", type=["pdf"])
    if uploaded_pdf:
        # Use session state to avoid re-ingesting on every rerender
        if uploaded_pdf.name not in st.session_state.get("ingested_pdfs", []):
            tmp_path = f"/tmp/{uploaded_pdf.name}"
            with open(tmp_path, "wb") as f:
                f.write(uploaded_pdf.read())
            chunks = ingest_single_pdf(tmp_path)
            if chunks:
                st.success(f" Ingested {uploaded_pdf.name} ({chunks} chunks)")
                # Track which PDFs have already been ingested
                if "ingested_pdfs" not in st.session_state:
                    st.session_state["ingested_pdfs"] = []
                st.session_state["ingested_pdfs"].append(uploaded_pdf.name)
            else:
                st.error(" Failed to ingest PDF.")
        else:
            st.info(f"'{uploaded_pdf.name}' already loaded into memory.")

    st.markdown("---")

    # -- Example Queries --
    st.header(" Example Queries")
    examples = [
        "My 70-year-old father has CKD. Book a nephrologist and summarize latest treatments.",
        "Retrieve history for patient P001",
        "What are the latest guidelines for managing hypertension?",
        "Book a cardiology appointment for John Smith",
        "Summarize Anjali's recent visit",
        "What were David's diabetes follow-up notes?"
    ]
    for ex in examples:
        if st.button(ex[:50] + "...", key=ex):
            st.session_state["prefill"] = ex

# ── Chat Interface ────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

# Render chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Handle prefill from example buttons or direct input
prefill = st.session_state.pop("prefill", "")
user_input = st.chat_input("Ask the healthcare assistant...") or prefill

if user_input:
    # Show user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner(" Agent is working..."):
            try:
                # Run the agent
                result = healthcare_agent.invoke({
                    "messages": [HumanMessage(content=user_input)],
                    "patient_id": patient_id
                })

                # Show task plan if available
                if result.get("task_plan"):
                    with st.expander(" Task Plan", expanded=True):
                        for task in result["task_plan"]:
                            if task.strip():
                                st.markdown(f"- {task}")

                # Show intermediate tool messages if available
                tool_messages = [
                    m for m in result.get("messages", [])
                    if hasattr(m, "name") and m.name  # tool response messages have .name
                ]
                if tool_messages:
                    with st.expander(" Agent Reasoning Steps", expanded=False):
                        for tm in tool_messages:
                            st.markdown(f"**Tool `{tm.name}`:**")
                            st.code(tm.content[:500])  # truncate long outputs

                # Display final response
                final = result.get("final_response") or "I processed your request."

            except Exception as e:
                final = f" Something went wrong: {str(e)}"

            st.markdown(final)

    # Append assistant response to history
    st.session_state.messages.append({"role": "assistant", "content": final})
