"""
In-memory mock database for the `support_tickets` table.

This is intentionally simple (a dict keyed by ticket number) so the project
can run with zero external dependencies. Swap this module out for a real
SQLite/Postgres implementation later without touching any agent code, since
every other module only talks to the functions defined here.
"""

import random
import threading
from datetime import datetime
from typing import Optional

from config import TICKET_NUMBER_DIGITS, TICKET_STATUSES

_lock = threading.Lock()

# ticket_number (str) -> ticket record (dict)
_support_tickets: dict[str, dict] = {}


def _generate_ticket_number() -> str:
    """Generate a unique N-digit ticket number not already in use."""
    low = 10 ** (TICKET_NUMBER_DIGITS - 1)
    high = (10 ** TICKET_NUMBER_DIGITS) - 1
    while True:
        candidate = str(random.randint(low, high))
        if candidate not in _support_tickets:
            return candidate


def create_ticket(customer_message: str, customer_name: Optional[str] = None) -> dict:
    """
    Insert a new unresolved ticket into support_tickets.

    Returns the created ticket record.
    """
    with _lock:
        ticket_number = _generate_ticket_number()
        record = {
            "ticket_number": ticket_number,
            "customer_name": customer_name or "Unknown",
            "original_message": customer_message,
            "status": "Open",
            "created_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "updated_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        }
        _support_tickets[ticket_number] = record
        return dict(record)


def get_ticket(ticket_number: str) -> Optional[dict]:
    """Look up a ticket by number. Returns None if not found."""
    with _lock:
        record = _support_tickets.get(ticket_number.strip())
        return dict(record) if record else None


def update_ticket_status(ticket_number: str, status: str) -> Optional[dict]:
    """Update a ticket's status. Returns the updated record, or None if not found."""
    if status not in TICKET_STATUSES:
        raise ValueError(f"Invalid status '{status}'. Must be one of {TICKET_STATUSES}")
    with _lock:
        record = _support_tickets.get(ticket_number.strip())
        if not record:
            return None
        record["status"] = status
        record["updated_at"] = datetime.utcnow().isoformat(timespec="seconds") + "Z"
        return dict(record)


def list_tickets() -> list[dict]:
    """Return all tickets, most recently created first (used by the UI)."""
    with _lock:
        return sorted(
            _support_tickets.values(), key=lambda r: r["created_at"], reverse=True
        )


def seed_demo_data() -> None:
    """Populate a few sample tickets so the Query Handler has something to find on a fresh run."""
    if _support_tickets:
        return
    demo = [
        ("Debit card replacement delayed", "Resolved"),
        ("Unable to reset net banking password", "In Progress"),
        ("Duplicate charge on statement", "Open"),
    ]
    for msg, status in demo:
        rec = create_ticket(msg, customer_name="Demo Customer")
        update_ticket_status(rec["ticket_number"], status)


def reset_db() -> None:
    """Clear all tickets. Useful for tests and Streamlit 'reset demo' button."""
    with _lock:
        _support_tickets.clear()
