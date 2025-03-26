import os
from langchain_core.prompts import ChatPromptTemplate
from .state import AgentState
from .utils.PROMPT import AGENT_PROMPT

def generator_prompt_template():
    prompt_template = ChatPromptTemplate([
        ("system", "%s"%AGENT_PROMPT),
        ("system", "你是一個數據分析師，請依照{question}回答分析數據，並詳細回答。若有需要生成圖表，請根據情況提供，不要提供額外的資訊，但圖表只是輔助，若不需要圖表，請專注於數據分析的內容，並詳盡解釋。若圖表需要生成，請依照以下規範處理："),
        # ("system", "若需要生成圖表，儲存的路徑：{chartTempPath}，檔名：{chartTempName}。"),
        # ("system", "在 matplotlib 程式裡面，要加上支援繁體中文字的程式碼，字型路徑：{font_file}。"),
        ("system", "表格檔案路徑，csv_path = {csv_path}，各欄位資訊可以參考 {info}"),
        ("system", "不需要將圖表顯示出來，請依照存檔資訊將它存下來。"),
        ("system", "不需要提及圖表儲存的資訊，只要將分析結果呈現出來。"),
        ("system", "請專注於數據分析的解釋，並根據分析提供清晰、有深度的回答。"),
        ("ai", "{history}")
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

def data_info_read():
    text = ""
    with open('/Users/guffrey/local-chatbot-service/data/csv/info.txt', 'r') as file: 
        info = file.read() 
        text += info
    return text

def call_model(state: AgentState, config: dict):
    
    collect_history(state)

    prompt = config["configurable"]["prompt_template"]

    chain = prompt|config["configurable"]["model"]

    info_text = data_info_read()
    response = chain.invoke({"question": state["messages"], 
                             "history":state["historyMsg"], 
                             "csv_path":"/Users/guffrey/local-chatbot-service/data/csv/data.csv",
                             "info":info_text})
    return {"messages": [response]}