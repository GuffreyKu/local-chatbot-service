from langgraph.graph import START, END, StateGraph
from langgraph.prebuilt import ToolNode
from .src import call_model
from .state import AgentState
from .utils.tab_tool import code_model

tools = [code_model]

def route_tools(state) -> bool:
    messages = state["messages"]
    last_message = messages[-1]
    if last_message.tool_calls:
        return "continue"
    return "end"

class Graph:
    def __init__(self):
        self.workflow = StateGraph(AgentState)
       
    def compile(self, memory):

        self.workflow.add_node("agent", call_model)
        self.workflow.add_node("tools", ToolNode(tools))

        self.workflow.add_edge(START, "agent")
        self.workflow.add_conditional_edges("agent", route_tools, {"continue":"tools", "end":END})
        self.workflow.add_edge("tools", "agent")

        return self.workflow.compile(checkpointer=memory)