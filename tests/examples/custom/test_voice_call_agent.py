# Copyright 2026 Tenro.ai
# SPDX-License-Identifier: Apache-2.0

"""Advanced example: Voice call agent.

Tests an agent that handles phone calls with speech-to-text and text-to-speech.
"""

from tenro import Construct, Provider, link_agent, link_llm, link_tool
from tenro.simulate import llm, tool

# APPLICATION CODE


@link_tool("speech_to_text")
def speech_to_text(audio_chunk: bytes) -> str:
    """Convert speech audio to text."""
    return "I'd like to check my order status"


@link_tool("text_to_speech")
def text_to_speech(text: str) -> bytes:
    """Convert text to speech audio."""
    return b"audio_data"


@link_tool("lookup_order")
def lookup_order(phone_number: str) -> dict | None:
    """Look up order by customer phone number."""
    return {"order_id": "ORD-123", "status": "shipped", "eta": "Dec 26"}


@link_llm(Provider.OPENAI)
def generate_voice_response(context: dict, user_input: str) -> str:
    """Generate a natural conversational response."""
    import openai

    response = openai.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": f"Customer context: {context}"},
            {"role": "user", "content": user_input},
        ],
    )
    return response.choices[0].message.content


@link_agent("VoiceCallAgent")
class VoiceCallAgent:
    """Agent that handles voice calls end-to-end."""

    def run(self, phone_number: str, audio_input: bytes) -> bytes:
        """Run the voice call agent."""
        user_text = speech_to_text(audio_input)
        order = lookup_order(phone_number)
        response_text = generate_voice_response(order, user_text)
        return text_to_speech(response_text)


# TESTS


def test_voice_agent_handles_order_inquiry(construct: Construct):
    """Test voice agent answering order status question."""
    # Control what tools and LLMs return
    construct.simulate_tool(speech_to_text, result="Where's my order?")
    construct.simulate_tool(
        lookup_order,
        result={"order_id": "ORD-123", "status": "shipped", "eta": "Dec 26"},
    )
    construct.simulate_tool(text_to_speech, result=b"audio_response")
    construct.simulate_llm(
        Provider.OPENAI,
        target=generate_voice_response,
        response="Your order ORD-123 has shipped and will arrive by December 26th.",
    )

    # Run the agent
    VoiceCallAgent().run("+1234567890", b"audio_input")

    # Verify full pipeline ran
    tool.verify_many(speech_to_text, count=1)
    tool.verify_many(lookup_order, count=1)
    tool.verify_many(text_to_speech, count=1)
    llm.verify(Provider.OPENAI)


def test_voice_agent_handles_unknown_caller(construct: Construct):
    """Test voice agent with unrecognized phone number."""
    # Simulate unknown caller
    construct.simulate_tool(speech_to_text, result="Check my order")
    construct.simulate_tool(lookup_order, result=None)
    construct.simulate_tool(text_to_speech, result=b"audio")
    construct.simulate_llm(
        Provider.OPENAI,
        target=generate_voice_response,
        response="I couldn't find an order for this number. Can you provide your order ID?",
    )

    # Run the agent
    VoiceCallAgent().run("+9999999999", b"audio")

    # Verify graceful fallback
    tool.verify_many(lookup_order, count=1)
    llm.verify(output_contains="order ID")
