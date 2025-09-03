# ---- agents.py ----
import datetime
from langgraph.graph import END  
from dataclasses import asdict
from state import InquiryState  
from dotenv import load_dotenv
import openai
import os

# Load environment variables
load_dotenv()

# Initialize Azure OpenAI client
client = openai.AzureOpenAI(
    section...
)

def intake_agent(state: InquiryState) -> dict:
    """Handles the intake process and logs the request."""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    state.activity_log.append(f"Request received from {state.client_name} at {timestamp}.")
    return asdict(state)

def evaluation_agent(state: InquiryState) -> dict:
    """Uses an Azure OpenAI LLM to determine if the inquiry is approved."""
    try:
        prompt = (
            f"Analyze the following request for approval. Return only 'approved' or 'denied'.\n"
            f"Request: {state.request_details}"
        )
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that makes approval decisions."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=100
        )
        decision = response.choices[0].message.content.strip().lower()
        state.is_approved = "approved" in decision
        state.evaluation_notes = (
            "Request meets approval criteria." if state.is_approved 
            else "Request does not meet approval criteria."
        )
        state.activity_log.append("Evaluation outcome: " + state.evaluation_notes)
    except Exception as e:
        state.activity_log.append(f"Evaluation error: {str(e)}")
        state.is_approved = False
        state.evaluation_notes = "Error occurred during evaluation."
    
    return asdict(state)

def scheduling_agent(state: InquiryState) -> dict:
    """Schedules a meeting if the request is approved."""
    if state.is_approved:
        meeting_time = datetime.datetime.now() + datetime.timedelta(days=1)
        state.appointment_time = meeting_time.strftime("%Y-%m-%d %H:%M:%S")
        state.activity_log.append("Meeting set for " + state.appointment_time)
    else:
        state.activity_log.append("No appointment set as request was denied.")
    return asdict(state)

def crm_update_agent(state: InquiryState) -> dict:
    """Handles CRM update and properly ends the workflow."""
    state.crm_log = "CRM updated with request details."
    state.activity_log.append("CRM updated successfully.")
    return asdict(state)
