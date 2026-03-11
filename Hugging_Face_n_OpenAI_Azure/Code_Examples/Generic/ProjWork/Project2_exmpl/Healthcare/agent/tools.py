import json
import random
import os
import requests
from datetime import datetime, timedelta
from langchain_core.tools import tool
from dotenv import load_dotenv

# ── Load env — works whether you run from project root or subfolder
load_dotenv()

# ── SerpAPI Key ───────────────────────────────────────────────────
SERPAPI_KEY = os.getenv("SERPAPI_KEY")

# ──────────────────────────────────────────────────────────────────
# DATA SOURCE TOGGLE
# Set USE_JSON_DB = True  → loads from data/patients.json
# Set USE_JSON_DB = False → uses hardcoded PATIENT_DB below
# ──────────────────────────────────────────────────────────────────
USE_JSON_DB = True

# ── Hardcoded Patient DB (fallback / dev mode) ────────────────────
HARDCODED_PATIENT_DB = {
    "P001": {
        "name": "John Smith",
        "age": 70,
        "gender": "Male",
        "contact": "+1-555-0101",
        "email": "john.smith@email.com",
        "blood_type": "A+",
        "conditions": ["Chronic Kidney Disease Stage 3", "Hypertension", "Type 2 Diabetes"],
        "medications": ["Lisinopril 10mg", "Amlodipine 5mg", "Metformin 500mg"],
        "last_visit": "2024-11-15",
        "next_appointment": "2025-03-10",
        "allergies": ["Penicillin", "Sulfa drugs"],
        "doctor": "Dr. Mehta (Nephrologist)",
        "visit_history": [
            {"date": "2024-11-15", "reason": "Routine CKD checkup", "notes": "Creatinine slightly elevated. Adjusted Lisinopril dosage."},
            {"date": "2024-08-20", "reason": "Hypertension follow-up", "notes": "BP stable at 130/80. Continue current meds."},
            {"date": "2024-05-10", "reason": "Diabetes management", "notes": "HbA1c at 7.2%. Diet counseling provided."}
        ],
        "lab_results": {
            "creatinine": "2.1 mg/dL",
            "eGFR": "42 mL/min",
            "HbA1c": "7.2%",
            "blood_pressure": "132/82 mmHg"
        },
        "emergency_contact": {"name": "Mary Smith", "relation": "Spouse", "phone": "+1-555-0102"}
    },
    "P002": {
        "name": "Sarah Johnson",
        "age": 45,
        "gender": "Female",
        "contact": "+1-555-0201",
        "email": "sarah.johnson@email.com",
        "blood_type": "O-",
        "conditions": ["Asthma", "Anxiety Disorder", "Hypothyroidism"],
        "medications": ["Albuterol inhaler", "Levothyroxine 50mcg", "Sertraline 25mg"],
        "last_visit": "2025-01-08",
        "next_appointment": "2025-04-08",
        "allergies": ["Aspirin", "NSAIDs"],
        "doctor": "Dr. Sharma (Pulmonologist)",
        "visit_history": [
            {"date": "2025-01-08", "reason": "Asthma flare-up", "notes": "Prescribed short course of oral steroids. Inhaler technique reviewed."},
            {"date": "2024-10-15", "reason": "Thyroid follow-up", "notes": "TSH within normal range. Continue Levothyroxine."},
            {"date": "2024-07-22", "reason": "Anxiety management", "notes": "Sertraline dosage reviewed. CBT referral given."}
        ],
        "lab_results": {
            "TSH": "2.4 mIU/L",
            "T4": "1.1 ng/dL",
            "peak_flow": "380 L/min",
            "blood_pressure": "118/74 mmHg"
        },
        "emergency_contact": {"name": "Tom Johnson", "relation": "Husband", "phone": "+1-555-0202"}
    },
    "P003": {
        "name": "Robert Lee",
        "age": 58,
        "gender": "Male",
        "contact": "+1-555-0301",
        "email": "robert.lee@email.com",
        "blood_type": "B+",
        "conditions": ["Coronary Artery Disease", "High Cholesterol", "Obesity"],
        "medications": ["Atorvastatin 40mg", "Aspirin 81mg", "Metoprolol 25mg"],
        "last_visit": "2025-02-01",
        "next_appointment": "2025-05-01",
        "allergies": ["Contrast dye"],
        "doctor": "Dr. Nair (Cardiologist)",
        "visit_history": [
            {"date": "2025-02-01", "reason": "Post-stent follow-up", "notes": "Recovery good. Continue dual antiplatelet therapy for 12 months."},
            {"date": "2024-11-10", "reason": "Chest pain evaluation", "notes": "Stress test performed. Referred for coronary angiography."},
            {"date": "2024-09-05", "reason": "Cholesterol review", "notes": "LDL still high at 145. Increased Atorvastatin to 40mg."}
        ],
        "lab_results": {
            "LDL": "118 mg/dL",
            "HDL": "42 mg/dL",
            "triglycerides": "210 mg/dL",
            "blood_pressure": "138/88 mmHg",
            "BMI": "31.2"
        },
        "emergency_contact": {"name": "Linda Lee", "relation": "Wife", "phone": "+1-555-0302"}
    },
    "P004": {
        "name": "Priya Patel",
        "age": 32,
        "gender": "Female",
        "contact": "+1-555-0401",
        "email": "priya.patel@email.com",
        "blood_type": "AB+",
        "conditions": ["Gestational Diabetes (resolved)", "Iron Deficiency Anemia", "Migraine"],
        "medications": ["Ferrous Sulfate 325mg", "Sumatriptan 50mg (as needed)", "Folic Acid 400mcg"],
        "last_visit": "2025-01-20",
        "next_appointment": "2025-03-20",
        "allergies": ["Latex"],
        "doctor": "Dr. Singh (General Practitioner)",
        "visit_history": [
            {"date": "2025-01-20", "reason": "Anemia follow-up", "notes": "Hemoglobin improved to 11.2. Continue iron supplements."},
            {"date": "2024-12-05", "reason": "Migraine management", "notes": "Frequency increased. Sumatriptan prescribed for acute episodes."},
            {"date": "2024-09-15", "reason": "Post-partum checkup", "notes": "Gestational diabetes resolved. Blood sugar normal. Monitor annually."}
        ],
        "lab_results": {
            "hemoglobin": "11.2 g/dL",
            "ferritin": "8 ng/mL",
            "fasting_glucose": "92 mg/dL",
            "blood_pressure": "110/70 mmHg"
        },
        "emergency_contact": {"name": "Raj Patel", "relation": "Husband", "phone": "+1-555-0402"}
    },
    "P005": {
        "name": "Michael Brown",
        "age": 65,
        "gender": "Male",
        "contact": "+1-555-0501",
        "email": "michael.brown@email.com",
        "blood_type": "A-",
        "conditions": ["COPD Stage 2", "Osteoarthritis", "Depression"],
        "medications": ["Tiotropium inhaler", "Fluticasone/Salmeterol", "Celecoxib 200mg", "Escitalopram 10mg"],
        "last_visit": "2025-02-10",
        "next_appointment": "2025-04-10",
        "allergies": ["Codeine", "Tramadol"],
        "doctor": "Dr. Kumar (Pulmonologist)",
        "visit_history": [
            {"date": "2025-02-10", "reason": "COPD exacerbation", "notes": "Antibiotics prescribed. Oxygen saturation 94%. Pulmonary rehab recommended."},
            {"date": "2024-12-18", "reason": "Knee pain (osteoarthritis)", "notes": "X-ray shows moderate joint space narrowing. Physio referral given."},
            {"date": "2024-10-01", "reason": "Depression screening", "notes": "PHQ-9 score 12. Started Escitalopram. Follow-up in 6 weeks."}
        ],
        "lab_results": {
            "FEV1": "58% predicted",
            "oxygen_saturation": "94%",
            "blood_pressure": "125/78 mmHg",
            "BMI": "26.8"
        },
        "emergency_contact": {"name": "Carol Brown", "relation": "Daughter", "phone": "+1-555-0502"}
    }
}

# ── Load Patient DB based on toggle ──────────────────────────────
def _load_patient_db() -> dict:
    if USE_JSON_DB:
        db_path = os.path.join(os.path.dirname(__file__), "../data/patients.json")
        if os.path.exists(db_path):
            with open(db_path, "r") as f:
                print("Loaded patient DB from patients.json")
                return json.load(f)
        else:
            print("patients.json not found — falling back to hardcoded DB")
            return HARDCODED_PATIENT_DB
    else:
        print("Using hardcoded patient DB")
        return HARDCODED_PATIENT_DB

PATIENT_DB = _load_patient_db()

# ── Save helper (only works in JSON mode) ────────────────────────
def _save_patient_db():
    if USE_JSON_DB:
        db_path = os.path.join(os.path.dirname(__file__), "../data/patients.json")
        with open(db_path, "w") as f:
            json.dump(PATIENT_DB, f, indent=2)

# ── Doctor Schedules ──────────────────────────────────────────────
DOCTOR_SCHEDULES = {
    "nephrologist":    ["Dr. Mehta", "Dr. Patel"],
    "cardiologist":    ["Dr. Nair", "Dr. Sharma"],
    "pulmonologist":   ["Dr. Kumar", "Dr. Joshi"],
    "neurologist":     ["Dr. Rao", "Dr. Iyer"],
    "endocrinologist": ["Dr. Gupta", "Dr. Verma"],
    "general":         ["Dr. Singh", "Dr. Das"]
}

# ─────────────────────────────────────────────────────────────────
# TOOLS
# ─────────────────────────────────────────────────────────────────

@tool
def get_patient_history(patient_id: str) -> str:
    """Retrieve the full medical history for a patient using their patient ID (e.g. P001)."""
    patient = PATIENT_DB.get(patient_id)
    if not patient:
        available = list(PATIENT_DB.keys())
        return f"No patient found with ID: {patient_id}. Available IDs: {available}"
    return json.dumps(patient, indent=2)


@tool
def search_patient_by_name(name: str) -> str:
    """Search for a patient by their name and return their ID and basic info."""
    name_lower = name.lower()
    matches = []
    for pid, data in PATIENT_DB.items():
        if name_lower in data.get("name", "").lower():
            matches.append({
                "patient_id": pid,
                "name": data["name"],
                "age": data["age"],
                "conditions": data.get("conditions", []),
                "doctor": data.get("doctor", "N/A")
            })
    if not matches:
        return f"No patient found with name matching '{name}'."
    return json.dumps(matches, indent=2)


@tool
def book_appointment(specialty: str, patient_name: str) -> str:
    """Book a medical appointment for a patient with a specialist.
    Specialty options: nephrologist, cardiologist, pulmonologist, neurologist, endocrinologist, general.
    """
    doctors = DOCTOR_SCHEDULES.get(specialty.lower(), DOCTOR_SCHEDULES["general"])
    doctor = random.choice(doctors)
    slot_date = datetime.now() + timedelta(days=random.randint(1, 5))
    slot_time = f"{random.choice([9, 10, 11, 14, 15, 16])}:00"
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
    """Add or update a specific field in a patient's medical record.
    Changes are saved back to patients.json if JSON mode is active.
    """
    if patient_id not in PATIENT_DB:
        return f"Patient {patient_id} not found."
    PATIENT_DB[patient_id][field] = value
    _save_patient_db()
    mode = "patients.json" if USE_JSON_DB else "in-memory only"
    return f"Updated '{field}' for patient {patient_id} ({mode})."


@tool
def list_all_patients() -> str:
    """List all patients in the database with their ID, name, age, and primary conditions."""
    summary = []
    for pid, data in PATIENT_DB.items():
        summary.append({
            "patient_id": pid,
            "name": data.get("name"),
            "age": data.get("age"),
            "conditions": data.get("conditions", []),
            "last_visit": data.get("last_visit")
        })
    return json.dumps(summary, indent=2)


@tool
def search_medical_info(query: str) -> str:
    """Search the web for up-to-date medical information, treatment guidelines,
    and disease info from sources like WHO, Mayo Clinic, CDC, and Medline."""
    if not SERPAPI_KEY:
        return "Search unavailable: SERPAPI_API_KEY not set in .env"
    url = "https://serpapi.com/search.json"
    params = {
        "q": f"{query} medical treatment guidelines",
        "api_key": SERPAPI_KEY,   # ← fixed: was SERPAPI_API_KEY (undefined variable)
        "num": 5,
    }
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        results = data.get("organic_results", [])
        if not results:
            return "No results found."
        snippets = [
            f"Source: {r.get('source', 'Unknown')}\n{r.get('snippet', '')}"
            for r in results[:3] if r.get("snippet")
        ]
        return "\n\n".join(snippets)
    except Exception as e:
        return f"Search failed: {str(e)}"
