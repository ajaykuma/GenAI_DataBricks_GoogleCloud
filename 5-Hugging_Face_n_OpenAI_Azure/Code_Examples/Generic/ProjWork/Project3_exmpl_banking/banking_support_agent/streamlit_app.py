"""
Streamlit dashboard for the Banking Support Multi-Agent System.

Tabs:
  1. Live Agent  - send a message, see classification + agent response
  2. Tickets     - browse the mock support_tickets table
  3. Logs        - prompt traces / debugging view + routing success rate
  4. Evaluation  - run the classification + routing eval suite on demand

The classifier works perfectly on its own (Positive Feedback is exactly right), 
but fails only when running inside Streamlit. This is almost certainly the async/event-loop conflict
- Streamlit's execution model interferes with the async call litellm makes under the hood.
"""
# import asyncio

# try:
#     asyncio.get_event_loop()
# except RuntimeError:
#     asyncio.set_event_loop(asyncio.new_event_loop())

# import nest_asyncio
# nest_asyncio.apply()

import streamlit as st
import pandas as pd

import database
import logger
from orchestrator import run_pipeline
from config import api_key, azure_endpoint, azure_deployment, api_version, DEFAULT_CUSTOMER_NAME

AZURE_CONFIGURED = all([azure_endpoint, api_key, api_version, azure_deployment])

st.set_page_config(page_title="Banking Support Multi-Agent System", page_icon="🏦", layout="wide")

# Seed some demo tickets so "Tickets" and the Query Handler have data on first run.
database.seed_demo_data()

st.title("🏦 Banking Customer Support — Multi-Agent Dashboard")
st.caption("Classifier Agent → Feedback Handler Agent / Query Handler Agent  |  built with CrewAI")

if not AZURE_CONFIGURED:
    st.warning(
        "Azure OpenAI is not fully configured. Set AZURE_OPENAI_ENDPOINT, API_KEY, "
        "AZURE_API_VERSION, and AZURE_DEPLOYMENT_NAME in a `.env` file before sending messages.",
        icon="⚠️",
    )

tab_live, tab_tickets, tab_logs, tab_eval = st.tabs(
    ["💬 Live Agent", "🎫 Tickets", "🧾 Logs & Debugging", "📊 Evaluation"]
)

# --- Tab 1: Live Agent -------------------------------------------------------
with tab_live:
    st.subheader("Simulate a customer message")

    col1, col2 = st.columns([2, 1])
    with col2:
        customer_name = st.text_input("Customer name", value=DEFAULT_CUSTOMER_NAME)
        st.markdown("**Try an example:**")
        examples = [
            "Thanks for sorting out my net banking login issue.",
            "My debit card replacement still hasn't arrived.",
            "Could you check the status of ticket 650932?",
        ]
        for ex in examples:
            if st.button(ex, use_container_width=True, key=f"ex_{ex}"):
                st.session_state["message_input"] = ex

    with col1:
        message = st.text_area(
            "Customer message",
            key="message_input",
            height=120,
            placeholder="e.g. Thanks for resolving my credit card issue.",
        )
        send = st.button("Send to Agents", type="primary")

    if send:
        if not AZURE_CONFIGURED:
            st.error("Cannot call the agents without Azure OpenAI configured.")
        else:
            with st.spinner("Routing through Classifier Agent..."):
                result = run_pipeline(message, customer_name=customer_name or DEFAULT_CUSTOMER_NAME)

            if result.success:
                st.success("Pipeline completed")
            else:
                st.error(f"Pipeline error: {result.error}")

            c1, c2, c3 = st.columns(3)
            c1.metric("Classification", result.classification)
            c2.metric("Ticket #", result.ticket_number or "—")
            c3.metric("Agent Path", "")
            st.caption(f"**Agent path:** {result.agent_path}")

            st.markdown("**Response:**")
            st.info(result.response)

# --- Tab 2: Tickets -----------------------------------------------------------
with tab_tickets:
    st.subheader("support_tickets (mock DB)")
    col_a, col_b = st.columns([1, 5])
    with col_a:
        if st.button("Reset demo data"):
            database.reset_db()
            database.seed_demo_data()
            st.rerun()

    tickets = database.list_tickets()
    if tickets:
        st.dataframe(pd.DataFrame(tickets), use_container_width=True, hide_index=True)
    else:
        st.info("No tickets yet. Send negative feedback in the Live Agent tab to create one.")

    st.divider()
    st.markdown("**Look up a ticket manually**")
    lookup_num = st.text_input("Ticket number", key="lookup_num")
    if st.button("Look up"):
        ticket = database.get_ticket(lookup_num) if lookup_num else None
        if ticket:
            st.json(ticket)
        else:
            st.warning("No ticket found with that number.")

# --- Tab 3: Logs & Debugging --------------------------------------------------
with tab_logs:
    st.subheader("Agent interaction logs")
    col_a, col_b = st.columns([1, 5])
    with col_a:
        if st.button("Clear logs"):
            logger.clear_logs()
            st.rerun()

    st.metric("Overall success rate", f"{logger.success_rate()}%")

    logs = logger.get_logs()
    if logs:
        st.dataframe(pd.DataFrame(logs), use_container_width=True, hide_index=True)
        with st.expander("Raw trace (most recent first)"):
            for entry in logs:
                st.json(entry)
    else:
        st.info("No interactions logged yet.")

# --- Tab 4: Evaluation ---------------------------------------------------------
with tab_eval:
    st.subheader("Model & routing evaluation")
    st.caption(
        "Runs the labeled test set in evaluation.py against the live Classifier Agent "
        "and the full pipeline. Requires a configured Azure OpenAI connection (uses real LLM calls)."
    )

    if st.button("Run evaluation suite", type="primary", disabled=not AZURE_CONFIGURED):
        from evaluation import run_classification_eval, run_routing_eval

        with st.spinner("Running classification evaluation..."):
            class_report = run_classification_eval()
        with st.spinner("Running routing evaluation..."):
            routing_report = run_routing_eval()

        c1, c2 = st.columns(2)
        c1.metric("Classification accuracy", f"{class_report.accuracy}%", f"{class_report.correct}/{class_report.total}")
        c2.metric("Routing success rate", f"{routing_report['success_rate']}%", f"{routing_report['successes']}/{routing_report['total']}")

        if class_report.failures:
            st.markdown("**Classification failures:**")
            st.dataframe(pd.DataFrame(class_report.failures), use_container_width=True, hide_index=True)
        else:
            st.success("All classification test cases passed.")

        st.markdown("**Routing details:**")
        st.dataframe(pd.DataFrame(routing_report["details"]), use_container_width=True, hide_index=True)
    elif not AZURE_CONFIGURED:
        st.warning("Configure Azure OpenAI settings to run the evaluation suite.")
