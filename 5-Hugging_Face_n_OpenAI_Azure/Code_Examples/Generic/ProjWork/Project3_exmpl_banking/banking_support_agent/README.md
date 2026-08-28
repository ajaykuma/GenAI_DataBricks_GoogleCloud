# Banking Customer Support — Multi-Agent System

A CrewAI-based multi-agent assistant for banking customer support: classifies
incoming messages, routes them to the right specialist agent, manages a mock
support-ticket database, and exposes everything through a Streamlit dashboard.

## Architecture

```
User message
     │
     ▼
┌─────────────────────┐
│  Classifier Agent    │  -> "Positive Feedback" | "Negative Feedback" | "Query"
└─────────┬────────────┘
          │
   ┌──────┴───────────────────────┐
   ▼                               ▼
┌───────────────────────┐   ┌──────────────────────┐
│ Feedback Handler Agent │   │  Query Handler Agent  │
│  - Positive: thank-you │   │  - Extract ticket #    │
│  - Negative: creates a │   │  - Look up status in   │
│    ticket + apology    │   │    mock DB             │
└───────────┬────────────┘   └───────────┬───────────┘
            │                            │
            ▼                            ▼
       database.py (in-memory support_tickets table)
            │
            ▼
       logger.py (interaction/trace log for the UI)
```

Routing between the classifier and the two handler agents is a small Python
`if/elif` in `orchestrator.py`, not a CrewAI `Process` chain — the pipeline
branches on the classifier's output, and CrewAI's sequential/hierarchical
processes aren't a natural fit for conditional branching. Each branch still
does its actual work through a CrewAI `Agent`/`Task`/`Crew`.

## Project layout

```
banking_support_agent/
├── config.py              # env vars, model settings, shared constants
├── database.py            # mock in-memory support_tickets table
├── logger.py               # in-memory interaction/trace log
├── orchestrator.py         # classify -> route -> handle -> log
├── evaluation.py           # classification + routing eval harness
├── streamlit_app.py        # dashboard UI (4 tabs)
├── agents/
│   ├── llm.py               # shared CrewAI LLM factory
│   ├── classifier_agent.py  # Classifier Agent
│   ├── feedback_agent.py    # Feedback Handler Agent (positive + negative)
│   └── query_agent.py       # Query Handler Agent
├── requirements.txt
└── .env.example
```

## Setup

```bash
cd banking_support_agent
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# edit .env and set your env variables
```

## Run the dashboard

```bash
streamlit run streamlit_app.py
```

Tabs:
- **Live Agent** — type or pick an example message, see the classification,
  agent path, ticket number (if any), and generated response.
- **Tickets** — browse/reset the mock `support_tickets` table.
- **Logs & Debugging** — every interaction's trace (input, classification,
  agent path, response, success/failure) plus an overall success-rate metric.
- **Evaluation** — runs `evaluation.py`'s labeled test set against the live
  agents and reports classification accuracy + end-to-end routing success.

## Run the evaluation from the CLI

```bash
python3 evaluation.py
```

## Notes / next steps

- **Database**: currently in-memory (resets on restart). Swap `database.py`
  for a real SQLite/Postgres-backed module later — every other file only
  calls its public functions (`create_ticket`, `get_ticket`,
  `update_ticket_status`, `list_tickets`), so nothing else needs to change.
- **LangSmith**: not wired in yet. When you add it, the cleanest path is (a)
  set `LANGCHAIN_TRACING_V2=true` / `LANGCHAIN_API_KEY` in `.env`, and (b)
  reuse `evaluation.CLASSIFICATION_TEST_SET` as a LangSmith dataset so you
  get the same test cases in both the lightweight local eval and LangSmith's
  eval runners.
- **Model/provider**: change `OPENAI_MODEL` in `.env`, or swap the provider
  entirely in `agents/llm.py` (CrewAI's `LLM` class supports Anthropic,
  Ollama, etc. via the same `provider/model` string format).
