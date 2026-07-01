"""Tool (function-call) catalog grounded in the real Polish open-data APIs.

These OpenAI/Qwen-style function schemas mirror the endpoints already scraped by
scripts/ingest/ (GUS BDL, dane.gov.pl, ISAP). Grounding synthetic tool-use trajectories
in genuine APIs — instead of hallucinated tools — means the arguments the model learns to
emit map onto calls that actually exist, and the `meta` captured during ingestion gives us
real IDs/params to fill those arguments with.

Used by:
  - scripts/process/build_sft_qa.py  (--mode agentic) to attach `tools` to samples
  - scripts/eval/agentic_eval.py     to validate emitted tool calls

Each entry is a dict `{"type": "function", "function": {name, description, parameters}}`
where `parameters` is a JSON Schema (draft-2020-12 compatible) for the arguments object.
"""

from __future__ import annotations

# --- Individual tool schemas --------------------------------------------------

GUS_BDL_QUERY = {
    "type": "function",
    "function": {
        "name": "gus_bdl_query",
        "description": (
            "Zwraca wartość wskaźnika statystycznego z Banku Danych Lokalnych GUS "
            "(Bank Danych Lokalnych, Statistics Poland) dla danej zmiennej, jednostki "
            "terytorialnej i roku."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "subject_id": {
                    "type": "string",
                    "description": "Identyfikator dziedziny BDL, np. 'K11'.",
                },
                "variable_id": {
                    "type": "string",
                    "description": "Identyfikator zmiennej BDL.",
                },
                "year": {
                    "type": "integer",
                    "description": "Rok, którego dotyczy zapytanie.",
                },
                "unit": {
                    "type": "string",
                    "description": (
                        "Nazwa jednostki terytorialnej, np. 'POLSKA' lub nazwa "
                        "województwa."
                    ),
                },
            },
            "required": ["variable_id", "year"],
            "additionalProperties": False,
        },
    },
}

DANE_GOV_SEARCH = {
    "type": "function",
    "function": {
        "name": "dane_gov_search",
        "description": (
            "Wyszukuje zbiory danych w portalu dane.gov.pl (krajowy portal otwartych "
            "danych). Zwraca tytuły i opisy pasujących zbiorów."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Fraza wyszukiwania lub nazwa zbioru danych.",
                },
                "license": {
                    "type": "string",
                    "description": "Opcjonalny filtr licencji.",
                    "enum": ["cc0", "cc-by", "public-domain", "pddl", "odc-by"],
                },
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    },
}

ISAP_LOOKUP = {
    "type": "function",
    "function": {
        "name": "isap_lookup",
        "description": (
            "Pobiera metadane i treść polskiego aktu prawnego z ISAP (Internetowy "
            "System Aktów Prawnych) na podstawie wydawcy, roku i pozycji publikacji."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "publisher": {
                    "type": "string",
                    "description": "Wydawca: 'DU' = Dziennik Ustaw, 'MP' = Monitor Polski.",
                    "enum": ["DU", "MP"],
                },
                "year": {
                    "type": "integer",
                    "description": "Rok publikacji aktu.",
                },
                "position": {
                    "type": "integer",
                    "description": "Pozycja aktu w danym roczniku dziennika.",
                },
            },
            "required": ["publisher", "year", "position"],
            "additionalProperties": False,
        },
    },
}


# --- Registry + helpers -------------------------------------------------------

ALL_TOOLS = [GUS_BDL_QUERY, DANE_GOV_SEARCH, ISAP_LOOKUP]
_BY_NAME = {t["function"]["name"]: t for t in ALL_TOOLS}

# Which tool grounds which ingest `source` (see scripts/ingest/).
SOURCE_TO_TOOL = {
    "gus_bdl": "gus_bdl_query",
    "dane.gov.pl": "dane_gov_search",
    "isap": "isap_lookup",
}


def get_tool(name: str) -> dict:
    """Return the full tool schema for `name` (raises KeyError if unknown)."""
    return _BY_NAME[name]


def tool_for_source(source: str) -> str | None:
    """Map an ingest `source` to its grounding tool name, or None if unmapped."""
    return SOURCE_TO_TOOL.get(source)


def parameters_schema(name: str) -> dict:
    """Return just the JSON Schema for a tool's arguments object."""
    return _BY_NAME[name]["function"]["parameters"]
