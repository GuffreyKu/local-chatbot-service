import os
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import MemorySaver
from graph.flow import Graph
from graph.utils.pdf_embedding import multi_pdf_embedding

llm = ChatOllama(
        model="deepseek-r1:14b",
        temperature=0
    )

pdf_list = ["../data/pdf/"+pdf_path for pdf_path in os.listdir("../data/pdf/")]

vector_store = multi_pdf_embedding(pdf_list)

config = {"configurable": {"thread_id": "1",
                           "model": llm,
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