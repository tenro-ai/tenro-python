# Copyright 2026 Tenro.ai
# SPDX-License-Identifier: Apache-2.0

"""Shared tools for example applications.

These tools represent common business logic that can be used across
different AI framework implementations. All tools are decorated with
@link_tool for Tenro simulation support.
"""

from __future__ import annotations

from tenro import link_tool

# =============================================================================
# BUSINESS LOGIC TOOLS (from shared/tools.py)
# =============================================================================


@link_tool
def search_knowledge_base(query: str) -> list[dict]:
    """Search the company knowledge base for relevant articles.

    In production, this would query a vector database or search API.

    Args:
        query: Search query string.

    Returns:
        List of matching articles with title and content.
    """
    # Production implementation would query a real knowledge base
    return [{"title": "Refund Policy", "content": "Full refunds within 30 days."}]


@link_tool
def fetch_documents(topic: str) -> list[dict]:
    """Fetch documents from a document store.

    In production, this would query a document database or API.

    Args:
        topic: Topic to fetch documents for.

    Returns:
        List of documents with id and text content.
    """
    # Production implementation would query a real document store
    return [
        {"id": "doc1", "text": f"Introduction to {topic}..."},
        {"id": "doc2", "text": f"Advanced concepts in {topic}..."},
    ]


@link_tool
def summarize_email(email_body: str) -> str:
    """Summarize an email body.

    Args:
        email_body: The full email body text.

    Returns:
        A brief summary of the email.
    """
    # Production would call an LLM or summarization service
    return f"Summary of: {email_body[:50]}..."


@link_tool
def fetch_emails(folder: str, limit: int = 10) -> list[dict]:
    """Fetch emails from a mail server.

    Args:
        folder: Mail folder to fetch from (e.g., 'inbox').
        limit: Maximum number of emails to fetch.

    Returns:
        List of email dictionaries with subject and body.
    """
    # Production would connect to IMAP/Exchange
    return [{"subject": f"Email {i}", "body": f"Body of email {i}"} for i in range(min(limit, 3))]


# =============================================================================
# PATTERN TOOLS (from patterns/myapp/agents.py)
# =============================================================================


@link_tool
def get_weather(city: str) -> dict:
    """Get current weather for a city."""
    # Production: call weather API
    return {"temp": 70, "condition": "sunny"}


@link_tool
def search_database(query: str) -> list[dict]:
    """Search the database for records."""
    # Production: query database
    return []


@link_tool
def call_api(endpoint: str) -> dict:
    """Call an external API endpoint."""
    # Production: make HTTP request
    return {"status": "ok"}


@link_tool("search_documents")
def search_documents(query: str) -> list[str]:
    """Search for documents matching query."""
    # Real implementation calls vector DB
    return []


@link_tool("fetch_weather")
def fetch_weather(city: str) -> dict:
    """Fetch current weather for a city."""
    # Real implementation calls weather API
    return {}


@link_tool("search")
def search(query: str, limit: int = 10) -> list[str]:
    """Search for items matching query."""
    return []


@link_tool("send_email")
def send_email(to: str, subject: str, body: str) -> bool:
    """Send an email."""
    return True


@link_tool
def check_cache(key: str) -> dict | None:
    """Check if key exists in cache."""
    return None


@link_tool
def fetch_from_api(key: str) -> dict:
    """Fetch data from external API."""
    return {"key": key, "value": "data"}


@link_tool("get_cached_data")
def get_cached_data(key: str) -> dict | None:
    """Get data from cache, returns None if not found."""
    return None


@link_tool("delete_all_records")
def delete_all_records() -> bool:
    """Dangerous operation that deletes all records."""
    return True


@link_tool
def validate_input(data: dict) -> bool:
    """Validate input data."""
    return True


@link_tool
def process_data(data: dict) -> dict:
    """Process the validated data."""
    return {"processed": True, **data}


@link_tool
def save_result(result: dict) -> str:
    """Save the processed result."""
    return "saved-123"
