"""
Central configuration for the Banking Support Multi-Agent System.

All environment-dependent settings (API keys, model names, etc.) live here
so the rest of the codebase never touches os.environ directly.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# --- LLM settings -----------------------------------------------------------
azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
api_key=os.getenv("API_KEY")
api_version=os.getenv("AZURE_API_VERSION")
azure_deployment=os.getenv("AZURE_DEPLOYMENT_NAME")

# --- App settings -------------------------------------------------------------
TICKET_NUMBER_DIGITS = 6
DEFAULT_CUSTOMER_NAME = "Valued Customer"

# Valid classification labels used across the pipeline. Keeping this as a
# single source of truth avoids typos causing silent routing failures.
LABEL_POSITIVE = "Positive Feedback"
LABEL_NEGATIVE = "Negative Feedback"
LABEL_QUERY = "Query"
VALID_LABELS = [LABEL_POSITIVE, LABEL_NEGATIVE, LABEL_QUERY]

# Ticket status lifecycle used by the mock database.
TICKET_STATUSES = ["Open", "In Progress", "Resolved", "Closed"]
