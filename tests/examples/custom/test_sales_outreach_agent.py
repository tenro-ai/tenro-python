# Copyright 2026 Tenro.ai
# SPDX-License-Identifier: Apache-2.0

"""Intermediate example: Sales outreach agent.

Tests an agent that qualifies leads and answers product questions.
"""

from tenro import Construct, Provider, link_agent, link_llm, link_tool
from tenro.simulate import llm, tool

# APPLICATION CODE


@link_tool("lookup_crm")
def lookup_crm(email: str) -> dict | None:
    """Look up customer in CRM."""
    return {"name": "John", "company": "Acme", "plan": "starter"}


@link_tool("get_pricing")
def get_pricing(plan: str) -> dict:
    """Get pricing information for a plan."""
    return {"plan": plan, "price": "$99/month", "features": ["..."]}


@link_llm(Provider.OPENAI)
def generate_pitch(customer_info: dict, pricing: dict, question: str) -> str:
    """Generate a personalized sales response."""
    import openai

    response = openai.chat.completions.create(
        model="gpt-4",
        messages=[
            {
                "role": "system",
                "content": f"Customer: {customer_info}\nPricing: {pricing}",
            },
            {"role": "user", "content": question},
        ],
    )
    return response.choices[0].message.content


@link_agent("SalesOutreachAgent")
class SalesOutreachAgent:
    """Agent that answers sales questions with personalized context."""

    def run(self, customer_email: str, question: str) -> str:
        """Run the sales outreach agent."""
        customer = lookup_crm(customer_email)
        pricing = get_pricing("enterprise")
        return generate_pitch(customer, pricing, question)


# TESTS


def test_sales_agent_upsells_existing_customer(construct: Construct):
    """Test that agent personalizes response for existing customers."""
    # Control what tools and LLMs return
    construct.simulate_tool(
        lookup_crm,
        result={"name": "John", "company": "Acme", "plan": "starter"},
    )
    construct.simulate_tool(
        get_pricing,
        result={"plan": "enterprise", "price": "$299/month", "features": ["SSO", "API"]},
    )
    construct.simulate_llm(
        Provider.OPENAI,
        response="Hi John! Upgrading to Enterprise would give you SSO and API access.",
    )

    # Run the agent
    SalesOutreachAgent().run("john@acme.com", "What's included in Enterprise?")

    # Verify full pipeline ran
    tool.verify_many(lookup_crm, count=1)
    tool.verify_many(get_pricing, count=1)
    llm.verify_many(Provider.OPENAI, at_least=1)


def test_sales_agent_handles_new_lead(construct: Construct):
    """Test agent behavior with unknown customer."""
    # Simulate new lead (no CRM data)
    construct.simulate_tool(lookup_crm, result=None)
    construct.simulate_tool(
        get_pricing,
        result={"plan": "enterprise", "price": "$299/month"},
    )
    construct.simulate_llm(
        Provider.OPENAI,
        response="Thanks for your interest! The Enterprise plan includes...",
    )

    # Run the agent
    SalesOutreachAgent().run("new@prospect.com", "Tell me about pricing")

    # Verify correct path taken
    tool.verify_many(lookup_crm, count=1)
    llm.verify_many(Provider.OPENAI, at_least=1)
