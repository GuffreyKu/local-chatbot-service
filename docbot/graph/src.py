import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM
from langchain.chains import ConversationalRetrievalChain
from .state import AgentState
from .utils.pdf_embedding import multi_pdf_embedding

def generator_prompt_template():
    prompt_template = ChatPromptTemplate([
        ("system", "你是一個公司內部文件查詢機器人，專門用於檢索和查詢內部文件。當用戶提問時，請務必優先查詢向量資料庫中的相關文件，以提供最準確和最有依據的回答。"),
        ("human", "Hello, who are you?"),
        ("ai", "我是你的公司內部文件查詢機器人。"),
        ("human", "Context: {context}"),        # Add context placeholder here
        ("human", "Chat History: {chat_history}"),
        ("human", "Question: {question}")
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
    prompt_template = generator_prompt_template()
    retriever = config["configurable"]["pdf_vector"].as_retriever(search_type="similarity", search_kwargs={"k": 4})  # Retrieves relevant content
    conversation_chain = ConversationalRetrievalChain.from_llm(config["configurable"]["model"], 
                                                               retriever,
                                                               combine_docs_chain_kwargs={"prompt": prompt_template})  # Creates a conversational retrieval chain

    question = state["messages"][-1].content
    result = conversation_chain.invoke({"question": question, "chat_history": chat_history})  # Queries the model

    return {"messages": [result.get("answer")]}
