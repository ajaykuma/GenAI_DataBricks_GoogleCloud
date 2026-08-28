"""
Query Handler Agent

Trigger: activated when the Classifier Agent labels the message as 'Query'.

- Extracts a ticket number from the user's message.
- Looks it up in the support_tickets mock database.
- Returns a formatted status message.

Ticket-number extraction is done with a regex first (fast, deterministic,
and doesn't burn an LLM call for something this structured); the LLM agent
is used only to phrase the final natural-language reply.
"""

import re
from crewai import Agent, Task, Crew, Process

from agents.llm import get_llm
from config import TICKET_NUMBER_DIGITS
import database

_query_agent = Agent(
    role="Ticket Status Assistant",
    goal="Clearly communicate a support ticket's current status to the customer.",
    backstory=(
        "You work on the status-lookup team at a digital bank. Customers ask you about "
        "existing tickets and you give a clear, factual, friendly one-line answer using "
        "the exact status provided to you -- never inventing or softening it."
    ),
    llm=get_llm(),
    verbose=False,
    allow_delegation=False,
)

_TICKET_NUM_PATTERN = re.compile(rf"\b\d{{{TICKET_NUMBER_DIGITS}}}\b")


def extract_ticket_number(message: str) -> str | None:
    """Extract the first N-digit ticket number found in the message, if any."""
    match = _TICKET_NUM_PATTERN.search(message)
    return match.group(0) if match else None


def handle_query(message: str) -> dict:
    """
    Resolve a customer query about ticket status.
    Returns {response, ticket_number, found}.
    """
    ticket_number = extract_ticket_number(message)

    if not ticket_number:
        return {
            "response": (
                "I couldn't find a ticket number in your message. Could you share the "
                f"{TICKET_NUMBER_DIGITS}-digit ticket number you'd like me to look up?"
            ),
            "ticket_number": None,
            "found": False,
        }

    ticket = database.get_ticket(ticket_number)

    if not ticket:
        return {
            "response": f"I couldn't find any ticket matching #{ticket_number}. Please double-check the number.",
            "ticket_number": ticket_number,
            "found": False,
        }

    task = Task(
        description=(
            f'A customer asked: "{message}"\n\n'
            f"Their ticket #{ticket_number} has status: {ticket['status']}.\n\n"
            f"Reply with exactly this format, filling in the values: "
            f"\"Your ticket #{ticket_number} is currently marked as: {ticket['status']}.\" "
            "Do not add any other text."
        ),
        expected_output="A single sentence status message in the exact required format.",
        agent=_query_agent,
    )
    crew = Crew(agents=[_query_agent], tasks=[task], process=Process.sequential, verbose=False)
    response_text = str(crew.kickoff()).strip()

    return {"response": response_text, "ticket_number": ticket_number, "found": True}
