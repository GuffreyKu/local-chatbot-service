import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import requests
from pydantic import Field
from typing import List, Optional
from langchain.llms.base import LLM  # Use the base LLM from langchain.llms.base
from langgraph.checkpoint.memory import MemorySaver
from graph.flow import Graph
from graph.utils.pdf_embedding import multi_pdf_embedding

class LocalLLM(LLM):
    url: str = Field(..., description="Local LLM endpoint URL")
    temperature: float = Field(1, description="Sampling temperature")

    @property
    def _llm_type(self) -> str:
        return "local_llm"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        data = {
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature
        }
        response = requests.post(self.url, json=data)
        
        if response.status_code != 200:
            raise Exception(f"Request failed with status {response.status_code}: {response.text}")
        response_json = response.json()
        # print(response_json)
        # Adjust the key based on your API's response structure.
        return response_json.get("choices")[0].get("message").get("content")

# Initialize the local LLM using your endpoint
local_llm = LocalLLM(url="http://localhost:8080/v1/chat/completions", temperature=1)


pdf_list = ["../data/pdf/"+pdf_path for pdf_path in os.listdir("../data/pdf/")]

vector_store = multi_pdf_embedding(pdf_list)

config = {"configurable": {"thread_id": "1",
                           "model": local_llm,
                           "pdf_vector": vector_store}
                           }

graph = Graph()
memory = MemorySaver()

if __name__ == '__main__':
    app = graph.compile(memory)
    
    # app.get_graph().draw_mermaid_png(output_file_path="graph.png")

    while True:
        user_input = input("user: ")
        
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break

        for chunk in app.stream(
            {"messages": [("human", user_input)]},
            stream_mode="values",
            config=config):
            chunk["messages"][-1].pretty_print()