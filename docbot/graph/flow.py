from langgraph.graph import START, END, StateGraph
from .src import call_model
from .state import AgentState


class Graph:
    def __init__(self):
        self.workflow = StateGraph(AgentState)
       
    def compile(self, memory):

        self.workflow.add_node("agent", call_model)
        # self.workflow.add_node("pdf", muti_pdf_model)

        self.workflow.add_edge(START, "agent")
        self.workflow.add_edge("agent", END)

        return self.workflow.compile(checkpointer=memory)