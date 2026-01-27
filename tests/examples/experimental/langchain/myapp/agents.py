# Copyright 2026 Tenro.ai
# SPDX-License-Identifier: Apache-2.0

"""LangChain-based agents using OpenAI.

These agents demonstrate LangChain patterns including:
- ChatPromptTemplate for structured prompts
- RunnableWithMessageHistory for conversation memory
- InMemoryChatMessageHistory for session-based history
- InMemoryVectorStore with retriever for RAG
- FakeEmbeddings for deterministic testing (no API calls)
"""

from __future__ import annotations

from examples.myapp import fetch_documents, search_knowledge_base
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.documents import Document
from langchain_core.embeddings import FakeEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import ChatOpenAI

from tenro import link_agent


@link_agent("CustomerSupportAgent")
class CustomerSupportAgent:
    """Answer customer questions using knowledge base and LangChain."""

    def run(self, question: str) -> str:
        """Process a customer support question.

        Uses InMemoryVectorStore with retriever for semantic search.

        Args:
            question: Customer's question.

        Returns:
            Support response.
        """
        articles = search_knowledge_base(question)
        docs = [Document(page_content=a["content"]) for a in articles]

        vectorstore = InMemoryVectorStore.from_documents(docs, FakeEmbeddings(size=1536))
        retriever = vectorstore.as_retriever()

        retrieved_docs = retriever.invoke(question)
        context = "\n".join(doc.page_content for doc in retrieved_docs)

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "Use this context to help customers:\n{context}"),
                ("user", "{question}"),
            ]
        )
        llm = ChatOpenAI(model="gpt-4o-mini")
        chain = prompt | llm

        response = chain.invoke({"context": context, "question": question})
        return response.content


@link_agent("RAGPipeline")
class RAGPipeline:
    """Answer questions using document retrieval and LangChain."""

    def run(self, question: str, topic: str) -> str:
        """Process a question with RAG.

        Uses InMemoryVectorStore with retriever for semantic search.

        Args:
            question: The question to answer.
            topic: Topic to retrieve documents for.

        Returns:
            Synthesized answer.
        """
        raw_docs = fetch_documents(topic)
        docs = [Document(page_content=d["text"]) for d in raw_docs]

        vectorstore = InMemoryVectorStore.from_documents(docs, FakeEmbeddings(size=1536))
        retriever = vectorstore.as_retriever()

        retrieved_docs = retriever.invoke(question)
        docs_text = "\n".join(doc.page_content for doc in retrieved_docs)

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "Answer based on these documents:\n{documents}"),
                ("user", "{question}"),
            ]
        )
        llm = ChatOpenAI(model="gpt-4o-mini")
        chain = prompt | llm

        response = chain.invoke({"documents": docs_text, "question": question})
        return response.content


@link_agent("ConversationAgent")
class ConversationAgent:
    """Handle multi-turn conversations with LangChain RunnableWithMessageHistory."""

    def run(self, user_messages: list[str]) -> list[str]:
        """Process multiple user messages maintaining context.

        Uses RunnableWithMessageHistory for memory management.

        Args:
            user_messages: List of user messages in order.

        Returns:
            List of assistant responses.
        """
        store: dict[str, InMemoryChatMessageHistory] = {}

        def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
            if session_id not in store:
                store[session_id] = InMemoryChatMessageHistory()
            return store[session_id]

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "You are a helpful coding assistant."),
                MessagesPlaceholder(variable_name="history"),
                ("human", "{input}"),
            ]
        )
        llm = ChatOpenAI(model="gpt-4o-mini")
        chain = prompt | llm

        chain_with_history = RunnableWithMessageHistory(
            chain,
            get_session_history,
            input_messages_key="input",
            history_messages_key="history",
        )

        config = {"configurable": {"session_id": "conversation-1"}}
        responses: list[str] = []

        for user_msg in user_messages:
            response = chain_with_history.invoke({"input": user_msg}, config=config)
            responses.append(response.content)

        return responses
