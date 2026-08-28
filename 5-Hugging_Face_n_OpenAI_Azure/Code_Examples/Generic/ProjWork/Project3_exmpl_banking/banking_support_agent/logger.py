"""
Lightweight in-memory logging for agent traces.

Captures: timestamp, user input, classification, agent path, response,
ticket actions, and success/failure — everything Part 2's "Logs and
Debugging View" needs. Kept in-process (a list) so the Streamlit UI can
render it directly without a separate log store.
"""

import threading
from datetime import datetime
from typing import Optional

_lock = threading.Lock()
_logs: list[dict] = []


def log_interaction(
    user_input: str,
    classification: str,
    agent_path: str,
    response: str,
    ticket_number: Optional[str] = None,
    success: bool = True,
    error: Optional[str] = None,
) -> dict:
    entry = {
        "timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "user_input": user_input,
        "classification": classification,
        "agent_path": agent_path,
        "response": response,
        "ticket_number": ticket_number,
        "success": success,
        "error": error,
    }
    with _lock:
        _logs.append(entry)
    return entry


def get_logs() -> list[dict]:
    with _lock:
        return list(reversed(_logs))  # most recent first


def clear_logs() -> None:
    with _lock:
        _logs.clear()


def success_rate() -> float:
    """Overall agent routing/response success rate, for the metrics view."""
    with _lock:
        if not _logs:
            return 0.0
        successes = sum(1 for l in _logs if l["success"])
        return round(100 * successes / len(_logs), 1)
