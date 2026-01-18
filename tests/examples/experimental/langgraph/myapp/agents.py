# Copyright 2026 Tenro.ai
# SPDX-License-Identifier: Apache-2.0

"""LangGraph-based agents using OpenAI.

These agents demonstrate LangGraph patterns including:
- MessagesState for built-in message handling
- add_messages reducer for message accumulation
- InMemorySaver checkpointer for persistence
- Message objects instead of string serialization
- InMemoryVectorStore with retriever for RAG
- FakeEmbeddings for deterministic testing (no API calls)
"""

from __future__ import annotations

from examples.myapp import fetch_documents, search_knowledge_base

from tenro import link_agent


@link_agent("CustomerSupportAgent")
class CustomerSupportAgent:
    """Answer customer questions using LangGraph workflow."""

    def run(self, question: str) -> str:
        """Process a customer support question.

        Uses InMemoryVectorStore with retriever node for semantic search.

        Args:
            question: Customer's question.

        Returns:
            Support response.
        """
        from typing import TypedDict

        from langchain_core.documents import Document
        from langchain_core.embeddings import FakeEmbeddings
        from langchain_core.vectorstores import InMemoryVectorStore
        from langgraph.graph import END, StateGraph

        class SupportState(TypedDict):
            question: str
            context: str
            response: str

        def retrieve_context(state: SupportState) -> SupportState:
            # Get articles and create vector store with retriever
            articles = search_knowledge_base(state["question"])
            docs = [Document(page_content=a["content"]) for a in articles]
            vectorstore = InMemoryVectorStore.from_documents(docs, FakeEmbeddings(size=1536))
            retriever = vectorstore.as_retriever()

            # Use retriever for semantic search
            retrieved_docs = retriever.invoke(state["question"])
            context = "\n".join(doc.page_content for doc in retrieved_docs)
            return {**state, "context": context}

        def generate_response(state: SupportState) -> SupportState:
            from langchain_core.messages import HumanMessage, SystemMessage
            from langchain_openai import ChatOpenAI

            llm = ChatOpenAI(model="gpt-4o-mini")
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


@link_agent("RAGPipeline")
class RAGPipeline:
    """Answer questions using LangGraph RAG workflow."""

    def run(self, question: str, topic: str) -> str:
        """Process a question with RAG.

        Uses InMemoryVectorStore with retriever node for semantic search.

        Args:
            question: User's question.
            topic: Topic to retrieve documents for.

        Returns:
            Synthesized answer.
        """
        from typing import TypedDict

        from langchain_core.documents import Document
        from langchain_core.embeddings import FakeEmbeddings
        from langchain_core.vectorstores import InMemoryVectorStore
        from langgraph.graph import END, StateGraph

        class RAGState(TypedDict):
            question: str
            topic: str
            retrieved_context: str
            answer: str

        def retriever_node(state: RAGState) -> RAGState:
            # Fetch documents and create vector store with retriever
            raw_docs = fetch_documents(state["topic"])
            docs = [Document(page_content=d["text"]) for d in raw_docs]
            vectorstore = InMemoryVectorStore.from_documents(docs, FakeEmbeddings(size=1536))
            retriever = vectorstore.as_retriever()

            # Use retriever for semantic search
            retrieved_docs = retriever.invoke(state["question"])
            context = "\n".join(doc.page_content for doc in retrieved_docs)
            return {**state, "retrieved_context": context}

        def synthesize_node(state: RAGState) -> RAGState:
            from langchain_core.messages import HumanMessage, SystemMessage
            from langchain_openai import ChatOpenAI

            llm = ChatOpenAI(model="gpt-4o-mini")
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


@link_agent("ConversationAgent")
class ConversationAgent:
    """Handle multi-turn conversations with LangGraph MessagesState and checkpointer."""

    def run(self, user_messages: list[str]) -> list[str]:
        """Process multiple user messages using MessagesState with checkpointer.

        Uses:
        - MessagesState for built-in message handling
        - InMemorySaver for conversation persistence
        - Message objects (HumanMessage, AIMessage)

        Args:
            user_messages: List of user messages in order.

        Returns:
            List of assistant responses.
        """
        from langchain_core.messages import HumanMessage, SystemMessage
        from langchain_openai import ChatOpenAI
        from langgraph.checkpoint.memory import MemorySaver
        from langgraph.graph import END, MessagesState, StateGraph

        def call_model(state: MessagesState) -> dict:
            llm = ChatOpenAI(model="gpt-4o-mini")
            # Prepend system message to conversation
            messages = [
                SystemMessage(content="You are a helpful coding assistant."),
                *state["messages"],
            ]
            response = llm.invoke(messages)
            # Return new message to be added via add_messages reducer
            return {"messages": [response]}

        # Build graph with MessagesState (has built-in add_messages reducer)
        graph = StateGraph(MessagesState)
        graph.add_node("model", call_model)
        graph.set_entry_point("model")
        graph.add_edge("model", END)

        # Compile with checkpointer for persistence
        checkpointer = MemorySaver()
        app = graph.compile(checkpointer=checkpointer)

        # Use thread_id for conversation continuity
        config = {"configurable": {"thread_id": "conversation-1"}}
        responses: list[str] = []

        for user_msg in user_messages:
            result = app.invoke(
                {"messages": [HumanMessage(content=user_msg)]},
                config=config,
            )
            # Last message is the assistant response
            responses.append(result["messages"][-1].content)

        return responses
