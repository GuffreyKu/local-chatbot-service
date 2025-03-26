from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import MemorySaver
from graph.src import generator_prompt_template
from graph.flow import Graph, tools


llm = ChatOllama(
        model="qwen2.5:14b",
        temperature=0
    ).bind_tools(tools)

prompt_template = generator_prompt_template()

config = {"configurable": {"thread_id": "1",
                           "prompt_template":prompt_template,
                           "model":llm}
                           }

graph = Graph()
memory = MemorySaver()

if __name__ == '__main__':
    app = graph.compile(memory)
    
    app.get_graph().draw_mermaid_png(output_file_path="graph.png")

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