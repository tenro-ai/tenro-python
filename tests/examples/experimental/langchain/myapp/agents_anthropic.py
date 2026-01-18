# Copyright 2026 Tenro.ai
# SPDX-License-Identifier: Apache-2.0

"""LangChain-based agents using Anthropic."""

from __future__ import annotations

from examples.myapp import fetch_documents, search_knowledge_base
from langchain_anthropic import ChatAnthropic
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.documents import Document
from langchain_core.embeddings import FakeEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.vectorstores import InMemoryVectorStore

from tenro import link_agent


@link_agent("CustomerSupportAgentAnthropic")
class CustomerSupportAgentAnthropic:
    """Answer customer questions using knowledge base and LangChain with Anthropic."""

    def run(self, question: str) -> str:
        """Process a customer support question."""
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
        llm = ChatAnthropic(model="claude-sonnet-4-20250514", api_key="test-key")
        chain = prompt | llm

        response = chain.invoke({"context": context, "question": question})
        return response.content


@link_agent("RAGPipelineAnthropic")
class RAGPipelineAnthropic:
    """Answer questions using document retrieval and LangChain with Anthropic."""

    def run(self, question: str, topic: str) -> str:
        """Process a question with RAG."""
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
        llm = ChatAnthropic(model="claude-sonnet-4-20250514", api_key="test-key")
        chain = prompt | llm

        response = chain.invoke({"documents": docs_text, "question": question})
        return response.content


@link_agent("ConversationAgentAnthropic")
class ConversationAgentAnthropic:
    """Handle multi-turn conversations with LangChain and Anthropic."""

    def run(self, user_messages: list[str]) -> list[str]:
        """Process multiple user messages maintaining context."""
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
        llm = ChatAnthropic(model="claude-sonnet-4-20250514", api_key="test-key")
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
