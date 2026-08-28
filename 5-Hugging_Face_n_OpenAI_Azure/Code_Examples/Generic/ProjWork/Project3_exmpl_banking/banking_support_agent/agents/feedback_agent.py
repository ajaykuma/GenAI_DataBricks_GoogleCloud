"""
Feedback Handler Agent

Trigger: activated when the Classifier Agent labels the message as
'Positive Feedback' or 'Negative Feedback'.

- Positive: generates a warm, personalized thank-you message.
- Negative: creates a new support ticket in the mock database and returns
  an empathetic message including the generated ticket number.
"""

from crewai import Agent, Task, Crew, Process

from agents.llm import get_llm
from config import LABEL_POSITIVE, DEFAULT_CUSTOMER_NAME
import database

_positive_agent = Agent(
    role="Customer Appreciation Specialist",
    goal="Write a short, warm, genuine thank-you message to a customer who left positive feedback.",
    backstory=(
        "You work on the goodwill team at a digital bank. Customers who take the time to "
        "say something nice deserve a reply that feels personal, not templated. You keep "
        "replies short (1-2 sentences), sincere, and never salesy."
    ),
    llm=get_llm(),
    verbose=False,
    allow_delegation=False,
)

_negative_agent = Agent(
    role="Customer Recovery Specialist",
    goal=(
        "Write a short, empathetic apology message to a customer who reported an unresolved "
        "problem, referencing the support ticket that has just been created for them."
    ),
    backstory=(
        "You work on the service recovery team at a digital bank. When something goes wrong "
        "for a customer, your job is to acknowledge the frustration sincerely and reassure "
        "them a real ticket now exists and a human will follow up. You never sound robotic "
        "or dismissive."
    ),
    llm=get_llm(),
    verbose=False,
    allow_delegation=False,
)


def handle_positive_feedback(message: str, customer_name: str = DEFAULT_CUSTOMER_NAME) -> dict:
    """Generate a personalized thank-you message. Returns {response, ticket_number}."""
    task = Task(
        description=(
            f'A customer named "{customer_name}" sent this positive feedback: "{message}"\n\n'
            f"Write a warm, personalized thank-you reply addressed to {customer_name}. "
            "Keep it to 1-2 sentences. Do not include a subject line or signature, just the message body."
        ),
        expected_output="A short thank-you message text, nothing else.",
        agent=_positive_agent,
    )
    crew = Crew(agents=[_positive_agent], tasks=[task], process=Process.sequential, verbose=False)
    response_text = str(crew.kickoff()).strip()
    return {"response": response_text, "ticket_number": None}


def handle_negative_feedback(message: str, customer_name: str = DEFAULT_CUSTOMER_NAME) -> dict:
    """
    Create a new unresolved ticket, then generate an empathetic reply
    referencing that ticket number. Returns {response, ticket_number}.
    """
    ticket = database.create_ticket(customer_message=message, customer_name=customer_name)
    ticket_number = ticket["ticket_number"]

    task = Task(
        description=(
            f'A customer named "{customer_name}" reported this unresolved problem: "{message}"\n\n'
            f"A support ticket #{ticket_number} has just been created for them. Write a short, "
            f"empathetic apology message that acknowledges the issue, mentions ticket #{ticket_number} "
            "explicitly, and reassures them the team will follow up shortly. Keep it to 2 sentences max."
        ),
        expected_output="A short empathetic apology message text mentioning the ticket number, nothing else.",
        agent=_negative_agent,
    )
    crew = Crew(agents=[_negative_agent], tasks=[task], process=Process.sequential, verbose=False)
    response_text = str(crew.kickoff()).strip()
    return {"response": response_text, "ticket_number": ticket_number}
