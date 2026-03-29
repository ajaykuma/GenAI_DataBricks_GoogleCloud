# ----main.py----
import streamlit as st
from state import InquiryState
from workflow import construct_graph, visualize_graph
from dataclasses import asdict
import matplotlib.pyplot as plt

def main():
    """Streamlit UI for inquiry processing system."""
    st.title("Automated Request Handling System")
    st.write("Enter the details below:")

    client_name = st.text_input("Name:")
    client_email = st.text_input("Email:")
    request_details = st.text_area("Request Details:")

    if st.button("Submit Request"):
        inquiry_state = InquiryState(
            client_name=client_name,
            client_email=client_email,
            request_details=request_details
        )
        state_dict = asdict(inquiry_state)
        
        process_graph = construct_graph()
        final_state = process_graph.invoke(state_dict)

        st.success("Request processed!")
        st.markdown("### Evaluation Outcome")
        st.write(final_state.get("evaluation_notes", ""))
        if final_state.get("is_approved"):
            st.markdown("### Meeting Scheduled")
            st.write(f"Meeting Time: {final_state.get('appointment_time', '')}")
        st.markdown("### CRM Update")
        st.write(final_state.get("crm_log", ""))
        st.markdown("### Activity Log")
        for entry in final_state.get("activity_log", []):
            st.write(f"- {entry}")

    # st.markdown("### Workflow Graph")
    graph_path = visualize_graph()
    plt.savefig('Workflow.png')
    # st.image(graph_path, caption="Process Workflow")

if __name__ == "__main__":
    main()