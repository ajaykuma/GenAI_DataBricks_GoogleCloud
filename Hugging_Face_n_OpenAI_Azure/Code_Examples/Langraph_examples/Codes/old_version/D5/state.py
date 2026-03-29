# ---- state.py ----
from dataclasses import dataclass, asdict, field
from typing import List

@dataclass
class InquiryState:
    """Defines the structure of inquiry state, tracking client details, process status, and logs."""
    client_name: str
    client_email: str
    request_details: str
    is_approved: bool = False
    evaluation_notes: str = ""
    appointment_time: str = ""
    crm_log: str = ""
    activity_log: List[str] = field(default_factory=list)