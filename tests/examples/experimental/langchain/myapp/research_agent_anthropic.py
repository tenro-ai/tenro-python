# Copyright 2026 Tenro.ai
# SPDX-License-Identifier: Apache-2.0

"""Research agent using LangChain's built-in tools with Anthropic."""

from langchain_anthropic import ChatAnthropic
from langchain_community.tools import DuckDuckGoSearchRun, WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_core.prompts import ChatPromptTemplate

from tenro import link_agent


@link_agent
class WebResearchAgentAnthropic:
    """Agent that uses DuckDuckGo and Wikipedia for research with Claude."""

    def run(self, question: str) -> str:
        """Research a topic using web search and Wikipedia.

        Args:
            question: The research question.

        Returns:
            Synthesized answer from research.
        """
        search = DuckDuckGoSearchRun()
        wiki = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())

        web_results = search.invoke(question)
        wiki_results = wiki.invoke(question)

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "Answer based on:\nWeb: {web}\nWikipedia: {wiki}"),
                ("user", "{question}"),
            ]
        )
        llm = ChatAnthropic(model="claude-sonnet-4-20250514", api_key="test-key")
        chain = prompt | llm

        response = chain.invoke({"web": web_results, "wiki": wiki_results, "question": question})
        return response.content
