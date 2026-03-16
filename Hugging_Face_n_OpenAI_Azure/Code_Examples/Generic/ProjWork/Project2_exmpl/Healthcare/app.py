import streamlit as st
import os
import tempfile
from datetime import datetime
from langchain_core.messages import HumanMessage
from agent.graph import healthcare_agent
from agent.memory import add_to_memory, ingest_single_pdf, save_vector_store, vector_store
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="🏥 Healthcare Assistant", layout="wide")
st.title(" Agentic Healthcare Assistant")
st.caption("Powered by GPT-4.1 · LangGraph · RAG")

# ── Helper: PDF ingestion ─────────────────────────────────────────
def _do_ingest(uploaded_pdf, patient_id, save_to_record, overwrite=False):
    """Handle PDF ingestion and optional record logging."""
    tmp_path = os.path.join(tempfile.gettempdir(), uploaded_pdf.name)
    with open(tmp_path, "wb") as f:
        f.write(uploaded_pdf.getvalue())

    chunks = ingest_single_pdf(tmp_path)

    if chunks:
        st.success(f" Ingested '{uploaded_pdf.name}' ({chunks} chunks)")
        if "ingested_pdfs" not in st.session_state:
            st.session_state["ingested_pdfs"] = []
        if uploaded_pdf.name not in st.session_state["ingested_pdfs"]:
            st.session_state["ingested_pdfs"].append(uploaded_pdf.name)

        if save_to_record and patient_id and patient_id.strip():
            from agent.tools import PATIENT_DB, _save_patient_db
            if patient_id in PATIENT_DB:
                if "documents" not in PATIENT_DB[patient_id]:
                    PATIENT_DB[patient_id]["documents"] = []
                if overwrite:
                    PATIENT_DB[patient_id]["documents"] = [
                        d for d in PATIENT_DB[patient_id]["documents"]
                        if d.get("filename") != uploaded_pdf.name
                    ]
                    st.info(" Old record entry removed — replacing with new.")
                PATIENT_DB[patient_id]["documents"].append({
                    "date": datetime.now().strftime("%Y-%m-%d"),
                    "time": datetime.now().strftime("%H:%M"),
                    "filename": uploaded_pdf.name,
                    "chunks": chunks,
                    "type": "uploaded_document"
                })
                _save_patient_db()
                st.info(f" Document logged to patient record ({patient_id})")
            else:
                st.warning(
                    f" Patient ID '{patient_id}' not found. "
                    f"Ingested to memory only."
                )
        elif save_to_record and not patient_id:
            st.warning(" No Patient ID entered. Ingested to memory only.")
    else:
        st.error(" Failed to ingest PDF.")

# ── Helper: Smart input enrichment ───────────────────────────────
def enrich_input(user_input: str, patient_id: str) -> str:
    patient_keywords = [
        "history", "record", "patient", "appointment", "book",
        "medication", "lab", "diagnosis", "visit", "schedule",
        "update", "allerg", "doctor", "treatment plan", "retrieve",
        "show me", "list all", "note", "pregnant", "result"
    ]
    query_lower = user_input.lower()
    is_patient_query = any(kw in query_lower for kw in patient_keywords)
    if is_patient_query and patient_id and patient_id.strip():
        return f"{user_input} (Patient ID: {patient_id})"
    return user_input

# ── Sidebar ───────────────────────────────────────────────────────
with st.sidebar:

    # -- Patient Context --
    st.header(" Patient Context")
    st.caption("Fill in for patient queries. Leave empty for general health questions.")
    patient_id = st.text_input("Patient ID (optional)", value="")

    if patient_id:
        st.info(f"Active patient: **{patient_id}**")
    else:
        st.warning("No patient selected — general health mode")

    st.markdown("---")

    # -- Add to Memory --
    st.header(" Add to Memory")
    new_memory = st.text_area("Add patient note:")

    save_to_record = st.checkbox(
        "Also save to patient record (patients.json)",
        value=False,
        key="save_note_checkbox",
        help="If checked, note will be permanently added to the patient's "
             "JSON record. Requires a valid Patient ID."
    )

    if st.button("Save to Memory") and new_memory:

        # ── Duplicate check in patients.json ─────────────────────
        duplicate_found = False
        if save_to_record and patient_id and patient_id.strip():
            from agent.tools import PATIENT_DB, _save_patient_db
            if patient_id in PATIENT_DB:
                existing_notes = PATIENT_DB[patient_id].get("notes", [])
                duplicate_found = any(
                    n.get("note", "").strip().lower() == new_memory.strip().lower()
                    for n in existing_notes
                )

        if duplicate_found:
            st.warning(
                " This exact note already exists in the patient record. "
                "Not saved to avoid duplicates."
            )
        else:
            # ── Duplicate check in FAISS ──────────────────────────
            existing_docs = vector_store.similarity_search(new_memory, k=3)
            faiss_duplicate = any(
                doc.page_content.strip().lower() == new_memory.strip().lower()
                for doc in existing_docs
            )

            if not faiss_duplicate:
                add_to_memory(new_memory, {
                    "patient_id": patient_id or "general",
                    "type": "manual_note"
                })
            else:
                st.info(
                    " Identical content already in memory — "
                    "skipping FAISS duplicate."
                )

            # ── Save to patients.json if checkbox ticked ──────────
            if save_to_record:
                if patient_id and patient_id.strip():
                    from agent.tools import PATIENT_DB, _save_patient_db
                    if patient_id in PATIENT_DB:
                        if "notes" not in PATIENT_DB[patient_id]:
                            PATIENT_DB[patient_id]["notes"] = []
                        PATIENT_DB[patient_id]["notes"].append({
                            "date": datetime.now().strftime("%Y-%m-%d"),
                            "time": datetime.now().strftime("%H:%M"),
                            "note": new_memory
                        })
                        _save_patient_db()
                        st.success(
                            f" Saved to memory AND patient record ({patient_id})!"
                        )
                    else:
                        st.warning(
                            f" Patient ID '{patient_id}' not found. "
                            f"Saved to memory only."
                        )
                else:
                    st.warning(
                        " No Patient ID entered. Saved to memory only."
                    )
            else:
                if not faiss_duplicate:
                    st.success(" Saved to memory!")

    st.markdown("---")

    # -- Upload PDF --
    st.header(" Upload Patient PDF")
    uploaded_pdf = st.file_uploader("Upload a report or record", type=["pdf"])

    save_pdf_to_record = st.checkbox(
        "Also save to patient record (patients.json)",
        value=False,
        key="save_pdf_checkbox",
        help="If checked, logs this document upload permanently to the "
             "patient's record. Requires a valid Patient ID."
    )

    if uploaded_pdf:
        # ── Check 1: already ingested this session ────────────────
        if uploaded_pdf.name in st.session_state.get("ingested_pdfs", []):
            st.info(f" '{uploaded_pdf.name}' already loaded this session.")
        else:
            # ── Check 2: already logged in patients.json ──────────
            already_in_record = False
            if save_pdf_to_record and patient_id and patient_id.strip():
                from agent.tools import PATIENT_DB, _save_patient_db
                if patient_id in PATIENT_DB:
                    existing_docs = PATIENT_DB[patient_id].get("documents", [])
                    already_in_record = any(
                        d.get("filename", "") == uploaded_pdf.name
                        for d in existing_docs
                    )

            if already_in_record:
                st.warning(
                    f" '{uploaded_pdf.name}' was previously uploaded "
                    f"for patient {patient_id}."
                )
                col1, col2 = st.columns(2)
                with col1:
                    overwrite = st.button(
                        " Re-ingest & Overwrite",
                        key="overwrite_btn"
                    )
                with col2:
                    skip = st.button(
                        " Skip — use existing",
                        key="skip_btn"
                    )
                if overwrite:
                    _do_ingest(
                        uploaded_pdf, patient_id,
                        save_pdf_to_record, overwrite=True
                    )
                elif skip:
                    st.info(" Skipped. Existing record retained.")
            else:
                # Fresh upload
                _do_ingest(
                    uploaded_pdf, patient_id,
                    save_pdf_to_record, overwrite=False
                )

    st.markdown("---")

    # -- Example Queries --
    st.header(" Example Queries")

    st.markdown("**👤 Patient queries:**")
    patient_examples = [
        "Retrieve history for this patient",
        "What medications is this patient on?",
        "Show lab results for this patient",
        "Book a nephrologist appointment",
        "List all patients in the system",
        "What is the doctor contact for this patient?",
    ]
    for ex in patient_examples:
        if st.button(ex, key=f"p_{ex}"):
            st.session_state["prefill"] = ex

    st.markdown("** General health questions:**")
    general_examples = [
        "What are the symptoms of diabetes?",
        "How is high blood pressure treated?",
        "What causes chronic kidney disease?",
        "Is it safe to take ibuprofen daily?",
        "What are the warning signs of a heart attack?",
        "What causes bronchitis?",
    ]
    for ex in general_examples:
        if st.button(ex, key=f"g_{ex}"):
            st.session_state["prefill"] = ex

# ── Chat Interface ────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

prefill = st.session_state.pop("prefill", "")
user_input = st.chat_input(
    "Ask anything — patient records or general health questions..."
) or prefill

if user_input:
    enriched_input = enrich_input(user_input, patient_id)

    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner(" Agent is working..."):
            try:
                # Build full conversation history
                history = []
                for msg in st.session_state.messages[:-1]:
                    if msg["role"] == "user":
                        history.append(HumanMessage(content=msg["content"]))
                    else:
                        history.append(
                            HumanMessage(content=f"[Assistant]: {msg['content']}")
                        )
                history.append(HumanMessage(content=enriched_input))

                result = healthcare_agent.invoke({
                    "messages": history,
                    "patient_id": patient_id or "none"
                })

                if result.get("task_plan"):
                    with st.expander(" Task Plan", expanded=False):
                        for task in result["task_plan"]:
                            if task.strip():
                                st.markdown(f"- {task}")

                tool_messages = [
                    m for m in result.get("messages", [])
                    if hasattr(m, "name") and m.name
                ]
                if tool_messages:
                    with st.expander(" Agent Reasoning Steps", expanded=False):
                        for tm in tool_messages:
                            st.markdown(f"**Tool: `{tm.name}`**")
                            st.code(tm.content[:500])

                final = result.get("final_response") or "I processed your request."

            except Exception as e:
                import traceback
                final = f" Error: {str(e)}"
                with st.expander(" Full Error", expanded=False):
                    st.code(traceback.format_exc())

            st.markdown(final)

    st.session_state.messages.append({"role": "assistant", "content": final})
