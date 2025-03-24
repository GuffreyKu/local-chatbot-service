import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM
from langchain.chains import ConversationalRetrievalChain
from .state import AgentState
from .utils.pdf_embedding import multi_pdf_embedding

def generator_prompt_template():
    prompt_template = ChatPromptTemplate([
    ("system", "You are a helpful assistant, you can read pdf and chat with user"),
    ("human", "Hello, who are you ?"),
    ("ai", "I'm ai assistant."),
    ("{history}"),
    ("human", "{question}"),
    ])
    return prompt_template

def collect_history(state: AgentState):
    if "historyMsg" not in state.keys():
        state["historyMsg"] = []
    for i, msg in enumerate(state["messages"][-2:]):
        if i%2 == 0:
            state["historyMsg"].append(("human", msg.content))
        else:
            state["historyMsg"].append(("ai", msg.content))
    return state

def call_model(state: AgentState, config: dict):
    
    collect_history(state)

    chat_history = []  # Stores previous conversation history

    retriever = config["configurable"]["pdf_vector"].as_retriever(search_type="similarity", search_kwargs={"k": 3})  # Retrieves relevant content
    conversation_chain = ConversationalRetrievalChain.from_llm(config["configurable"]["model"], retriever)  # Creates a conversational retrieval chain

    question = state["messages"][-1].content
    result = conversation_chain.invoke({"question": question, "chat_history": chat_history})  # Queries the model

    return {"messages": [result.get("answer")]}
