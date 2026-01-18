# Copyright 2026 Tenro.ai
# SPDX-License-Identifier: Apache-2.0

"""Intermediate example: Meeting notes agent.

Tests an agent that processes meeting recordings to extract insights.
"""

from tenro import Construct, Provider, link_agent, link_llm, link_tool
from tenro.simulate import llm, tool

# APPLICATION CODE


@link_tool("transcribe_audio")
def transcribe_audio(audio_url: str) -> str:
    """Transcribe meeting audio to text."""
    return "Meeting transcript..."


@link_tool("save_to_notion")
def save_to_notion(page_title: str, content: str) -> str:
    """Save content to Notion."""
    return "notion://page/123"


@link_llm(Provider.OPENAI)
def extract_summary(transcript: str) -> str:
    """Generate meeting summary."""
    import openai

    response = openai.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": f"Summarize this meeting:\n{transcript}"}],
    )
    return response.choices[0].message.content


@link_llm(Provider.OPENAI)
def extract_action_items(transcript: str) -> list[str]:
    """Extract action items from meeting."""
    import openai

    response = openai.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": f"List action items from:\n{transcript}"}],
    )
    return response.choices[0].message.content.split("\n")


@link_agent("MeetingNotesAgent")
class MeetingNotesAgent:
    """Agent that processes meetings and extracts insights."""

    def run(self, audio_url: str) -> dict:
        """Run the meeting notes agent."""
        transcript = transcribe_audio(audio_url)
        summary = extract_summary(transcript)
        action_items = extract_action_items(transcript)
        notion_url = save_to_notion("Meeting Notes", f"{summary}\n\n{action_items}")

        return {
            "summary": summary,
            "action_items": action_items,
            "notion_url": notion_url,
        }


# TESTS


def test_meeting_agent_processes_recording(construct: Construct):
    """Test full meeting processing pipeline."""
    # Control what tools and LLMs return
    construct.simulate_tool(
        transcribe_audio,
        result="John: Let's launch next week. Sarah: I'll prepare the docs.",
    )
    construct.simulate_tool(save_to_notion, result="notion://page/456")
    # Note: Multiple @link_llm functions need separate simulations or use HTTP interception
    construct.simulate_llm(
        Provider.OPENAI,
        responses=[
            "Team agreed to launch next week. Sarah will prepare documentation.",
            "- Sarah: Prepare launch docs\n- Team: Launch next week",
        ],
    )

    # Run the agent
    MeetingNotesAgent().run("https://zoom.us/recording/123")

    # Verify full pipeline ran
    tool.verify_many(transcribe_audio, count=1)
    tool.verify_many(save_to_notion, count=1)
    llm.verify_many(Provider.OPENAI, at_least=2)


def test_meeting_agent_handles_empty_transcript(construct: Construct):
    """Test agent with silent/empty recording."""
    # Simulate empty transcript
    construct.simulate_tool(transcribe_audio, result="")
    construct.simulate_tool(save_to_notion, result="notion://page/789")
    construct.simulate_llm(
        Provider.OPENAI,
        responses=["No content detected in recording.", "No action items."],
    )

    # Run the agent
    MeetingNotesAgent().run("https://zoom.us/recording/empty")

    # Verify graceful handling
    tool.verify_many(transcribe_audio, count=1)
    llm.verify_many(Provider.OPENAI, at_least=2)
