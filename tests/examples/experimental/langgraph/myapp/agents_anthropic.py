# Copyright 2026 Tenro.ai
# SPDX-License-Identifier: Apache-2.0

"""LangGraph-based agents using Anthropic.

These agents demonstrate LangGraph patterns with Anthropic's Claude models.
"""

from __future__ import annotations

from examples.myapp import fetch_documents, search_knowledge_base

from tenro import link_agent


@link_agent("CustomerSupportAgentAnthropic")
class CustomerSupportAgentAnthropic:
    """Answer customer questions using LangGraph with Anthropic."""

    def run(self, question: str) -> str:
        """Process a customer support question."""
        from typing import TypedDict

        from langchain_anthropic import ChatAnthropic
        from langchain_core.documents import Document
        from langchain_core.embeddings import FakeEmbeddings
        from langchain_core.messages import HumanMessage, SystemMessage
        from langchain_core.vectorstores import InMemoryVectorStore
        from langgraph.graph import END, StateGraph

        class SupportState(TypedDict):
            question: str
            context: str
            response: str

        def retrieve_context(state: SupportState) -> SupportState:
            articles = search_knowledge_base(state["question"])
            docs = [Document(page_content=a["content"]) for a in articles]
            vectorstore = InMemoryVectorStore.from_documents(docs, FakeEmbeddings(size=1536))
            retriever = vectorstore.as_retriever()
            retrieved_docs = retriever.invoke(state["question"])
            context = "\n".join(doc.page_content for doc in retrieved_docs)
            return {**state, "context": context}

        def generate_response(state: SupportState) -> SupportState:
            llm = ChatAnthropic(model="claude-3-5-sonnet-20241022")
            messages = [
                SystemMessage(content=f"Use this context to help:\n{state['context']}"),
                HumanMessage(content=state["question"]),
            ]
            response = llm.invoke(messages)
            return {**state, "response": response.content}

        graph = StateGraph(SupportState)
        graph.add_node("retrieve", retrieve_context)
        graph.add_node("respond", generate_response)
        graph.set_entry_point("retrieve")
        graph.add_edge("retrieve", "respond")
        graph.add_edge("respond", END)

        app = graph.compile()
        result = app.invoke({"question": question, "context": "", "response": ""})
        return result["response"]


@link_agent("RAGPipelineAnthropic")
class RAGPipelineAnthropic:
    """Answer questions using LangGraph RAG workflow with Anthropic."""

    def run(self, question: str, topic: str) -> str:
        """Process a question with RAG."""
        from typing import TypedDict

        from langchain_anthropic import ChatAnthropic
        from langchain_core.documents import Document
        from langchain_core.embeddings import FakeEmbeddings
        from langchain_core.messages import HumanMessage, SystemMessage
        from langchain_core.vectorstores import InMemoryVectorStore
        from langgraph.graph import END, StateGraph

        class RAGState(TypedDict):
            question: str
            topic: str
            retrieved_context: str
            answer: str

        def retriever_node(state: RAGState) -> RAGState:
            raw_docs = fetch_documents(state["topic"])
            docs = [Document(page_content=d["text"]) for d in raw_docs]
            vectorstore = InMemoryVectorStore.from_documents(docs, FakeEmbeddings(size=1536))
            retriever = vectorstore.as_retriever()
            retrieved_docs = retriever.invoke(state["question"])
            context = "\n".join(doc.page_content for doc in retrieved_docs)
            return {**state, "retrieved_context": context}

        def synthesize_node(state: RAGState) -> RAGState:
            llm = ChatAnthropic(model="claude-3-5-sonnet-20241022")
            messages = [
                SystemMessage(
                    content=f"Answer based on these documents:\n{state['retrieved_context']}"
                ),
                HumanMessage(content=state["question"]),
            ]
            response = llm.invoke(messages)
            return {**state, "answer": response.content}

        graph = StateGraph(RAGState)
        graph.add_node("retrieve", retriever_node)
        graph.add_node("synthesize", synthesize_node)
        graph.set_entry_point("retrieve")
        graph.add_edge("retrieve", "synthesize")
        graph.add_edge("synthesize", END)

        app = graph.compile()
        result = app.invoke(
            {
                "question": question,
                "topic": topic,
                "retrieved_context": "",
                "answer": "",
            }
        )
        return result["answer"]


@link_agent("ConversationAgentAnthropic")
class ConversationAgentAnthropic:
    """Handle multi-turn conversations with LangGraph using Anthropic."""

    def run(self, user_messages: list[str]) -> list[str]:
        """Process multiple user messages maintaining context."""
        from langchain_anthropic import ChatAnthropic
        from langchain_core.messages import HumanMessage, SystemMessage
        from langgraph.checkpoint.memory import MemorySaver
        from langgraph.graph import END, MessagesState, StateGraph

        def call_model(state: MessagesState) -> dict:
            llm = ChatAnthropic(model="claude-3-5-sonnet-20241022")
            messages = [
                SystemMessage(content="You are a helpful coding assistant."),
                *state["messages"],
            ]
            response = llm.invoke(messages)
            return {"messages": [response]}

        graph = StateGraph(MessagesState)
        graph.add_node("model", call_model)
        graph.set_entry_point("model")
        graph.add_edge("model", END)

        checkpointer = MemorySaver()
        app = graph.compile(checkpointer=checkpointer)

        config = {"configurable": {"thread_id": "conversation-1"}}
        responses: list[str] = []

        for user_msg in user_messages:
            result = app.invoke(
                {"messages": [HumanMessage(content=user_msg)]},
                config=config,
            )
            responses.append(result["messages"][-1].content)

        return responses
