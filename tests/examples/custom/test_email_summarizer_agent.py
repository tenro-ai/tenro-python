# Copyright 2026 Tenro.ai
# SPDX-License-Identifier: Apache-2.0

"""Basic example: Email summarizer agent.

Tests an agent that reads emails and generates summaries.
"""

from tenro import Construct, Provider, link_agent, link_llm, link_tool
from tenro.simulate import llm, tool

# APPLICATION CODE


@link_tool("fetch_emails")
def fetch_emails(folder: str, limit: int = 10) -> list[dict]:
    """Fetch emails from a folder."""
    return [{"subject": "Q4 Report", "body": "..."}]


@link_llm(Provider.OPENAI)
def summarize_email(email_body: str) -> str:
    """Generate a summary of an email."""
    import openai

    response = openai.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": f"Summarize this email:\n{email_body}"}],
    )
    return response.choices[0].message.content


@link_agent("EmailSummarizerAgent")
class EmailSummarizerAgent:
    """Agent that fetches and summarizes emails."""

    def run(self, folder: str) -> list[dict]:
        """Run the email summarizer agent."""
        emails = fetch_emails(folder, limit=5)
        return [
            {"subject": email["subject"], "summary": summarize_email(email["body"])}
            for email in emails
        ]


# TESTS


def test_email_agent_summarizes_inbox(construct: Construct):
    """Test that agent fetches and summarizes emails."""
    # Control what tools and LLMs return
    construct.simulate_tool(
        fetch_emails,
        result=[
            {"subject": "Q4 Report", "body": "Revenue increased 20%..."},
            {"subject": "Team Sync", "body": "Meeting moved to Thursday..."},
        ],
    )
    construct.simulate_llm(
        Provider.OPENAI,
        target=summarize_email,
        responses=["Q4 revenue up 20%.", "Team sync rescheduled to Thursday."],
    )

    # Run the agent
    EmailSummarizerAgent().run("inbox")

    # Verify behavior
    tool.verify_many(fetch_emails, count=1)
    llm.verify_many(Provider.OPENAI, count=2)


def test_email_agent_handles_empty_inbox(construct: Construct):
    """Test agent behavior with no emails."""
    # Simulate empty inbox
    construct.simulate_tool(fetch_emails, result=[])

    # Run the agent
    EmailSummarizerAgent().run("inbox")

    # Verify LLM was never called
    tool.verify_many(fetch_emails, count=1)
    llm.verify_never(Provider.OPENAI)
