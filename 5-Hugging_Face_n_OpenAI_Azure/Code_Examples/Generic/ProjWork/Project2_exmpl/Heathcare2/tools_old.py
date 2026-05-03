#pip install google-search-results
# All tools
#Each tool is a discrete capability. LangGraph routes to them based 
# on intent. Using @tool decorator makes them LangChain-compatible automatically.
import json, random
import os
from datetime import datetime, timedelta
from langchain_core.tools import tool
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.utilities import SerpAPIWrapper


# ── Mock Patient Database ────────────────────────
PATIENT_DB = {
    "P001": {
        "name": "John Smith",
        "age": 70,
        "conditions": ["Chronic Kidney Disease Stage 3", "Hypertension"],
        "medications": ["Lisinopril 10mg", "Amlodipine 5mg"],
        "last_visit": "2024-11-15",
        "allergies": ["Penicillin"],
        "doctor": "Dr. Mehta (Nephrologist)"
    }
}

# ── Mock Doctor Calendar ─────────────────────────
DOCTOR_SCHEDULES = {
    "nephrologist": ["Dr. Mehta", "Dr. Patel"],
    "cardiologist": ["Dr. Sharma", "Dr. Nair"],
    "general": ["Dr. Kumar", "Dr. Singh"]
}

@tool
def get_patient_history(patient_id: str) -> str:
    """Retrieve medical history for a patient by their ID."""
    patient = PATIENT_DB.get(patient_id)
    if not patient:
        return f"No patient found with ID: {patient_id}"
    return json.dumps(patient, indent=2)

@tool
def book_appointment(specialty: str, patient_name: str, preferred_date: str = "") -> str:
    """Book a medical appointment for a given specialty."""
    doctors = DOCTOR_SCHEDULES.get(specialty.lower(), DOCTOR_SCHEDULES["general"])
    doctor = random.choice(doctors)
    # Generate next available slot
    slot_date = datetime.now() + timedelta(days=random.randint(1, 5))
    slot_time = f"{random.choice([9,10,11,14,15,16])}:00"
    slot_str = slot_date.strftime("%A, %B %d, %Y")
    return (
        f"Appointment Confirmed!\n"
        f"  Doctor   : {doctor}\n"
        f"  Specialty: {specialty.title()}\n"
        f"  Patient  : {patient_name}\n"
        f"  Date/Time: {slot_str} at {slot_time}\n"
        f"  Location : City Medical Center, Room 204"
    )

@tool
def update_patient_record(patient_id: str, field: str, value: str) -> str:
    """Add or update a field in a patient's medical record."""
    if patient_id not in PATIENT_DB:
        return f"Patient {patient_id} not found."
    PATIENT_DB[patient_id][field] = value
    return f"Updated '{field}' for patient {patient_id}."

#@tool
# def search_medical_info(query: str) -> str:
#     """Search for up-to-date medical information using Wikipedia (proxy for Medline/WHO)."""
#     wiki = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper(top_k_results=2))
#     result = wiki.run(query)
#     return result[:2000]  # Truncate for context window


search = SerpAPIWrapper(serpapi_api_key=os.getenv("SERPAPI_API_KEY"))

@tool
def search_medical_info(query: str) -> str:
    """Search for up-to-date medical information from the web using Google Search.
    Use for disease information, treatment guidelines, drug interactions, and WHO/CDC updates.
    """
    # Bias the query toward trusted medical sources
    medical_query = f"{query} site:mayoclinic.org OR site:who.int OR site:medlineplus.gov OR site:cdc.gov"
    result = search.run(medical_query)
    return result[:3000]

#Or Use the SerpAPI Tool Directly (LangChain built-in)
#LangChain has a pre-built SerpAPI tool you can drop straight into the tools list:
# from langchain_community.tools import SerpAPIWrapper
# from langchain.agents import Tool

# search = SerpAPIWrapper()

# serpapi_tool = Tool(
#     name="search_medical_info",
#     func=search.run,
#     description="Searches the web for current medical information, disease treatments, drug info, and health guidelines from sources like WHO, Mayo Clinic, and Medline."
# )

