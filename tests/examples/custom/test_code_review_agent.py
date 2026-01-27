# Copyright 2026 Tenro.ai
# SPDX-License-Identifier: Apache-2.0

"""Intermediate example: Code review agent.

Tests an agent that reviews pull requests and suggests improvements.
"""

from __future__ import annotations

from tenro import Provider, link_agent, link_llm, link_tool
from tenro.simulate import llm, tool
from tenro.testing import tenro

# APPLICATION CODE


@link_tool("fetch_pr_diff")
def fetch_pr_diff(pr_url: str) -> str:
    """Fetch the diff from a pull request."""
    return "diff --git a/main.py..."


@link_tool("post_review_comment")
def post_review_comment(pr_url: str, comment: str) -> bool:
    """Post a review comment on the PR."""
    return True


@link_llm(Provider.OPENAI)
def analyze_code(diff: str) -> dict:
    """Analyze code changes for issues and improvements."""
    import openai

    response = openai.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": f"Review this code diff:\n{diff}"}],
    )
    return {"review": response.choices[0].message.content}


@link_agent("CodeReviewAgent")
class CodeReviewAgent:
    """Agent that reviews PRs and posts feedback."""

    def run(self, pr_url: str) -> dict:
        """Run the code review agent."""
        diff = fetch_pr_diff(pr_url)
        analysis = analyze_code(diff)
        post_review_comment(pr_url, analysis["review"])
        return analysis


# TESTS


@tenro
def test_code_review_agent_finds_security_issue():
    """Test that agent identifies code issues."""
    # Control what tools and LLMs return
    tool.simulate(
        fetch_pr_diff,
        result="+ def process(data):\n+     eval(data)  # dangerous!",
    )
    tool.simulate(post_review_comment, result=True)
    # Simulate at HTTP level - response becomes the message content
    llm.simulate(
        Provider.OPENAI,
        response="Security issue: Using eval() is dangerous. Use ast.literal_eval().",
    )

    # Run the agent
    CodeReviewAgent().run("https://github.com/org/repo/pull/123")

    # Verify issue was detected
    tool.verify_many(fetch_pr_diff, count=1)
    tool.verify_many(post_review_comment, count=1)
    llm.verify(output_contains="Security issue")


@tenro
def test_code_review_agent_approves_clean_code():
    """Test agent with well-formatted code."""
    # Simulate well-formatted PR
    tool.simulate(
        fetch_pr_diff,
        result="+ def add(a: int, b: int) -> int:\n+     return a + b",
    )
    tool.simulate(post_review_comment, result=True)
    llm.simulate(
        Provider.OPENAI,
        response="LGTM! Good implementation with type hints.",
    )

    # Run the agent
    CodeReviewAgent().run("https://github.com/org/repo/pull/456")

    # Verify approval given
    tool.verify_many(fetch_pr_diff, count=1)
    llm.verify(output_contains="LGTM")
