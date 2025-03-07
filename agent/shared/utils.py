from langgraph.graph.state import CompiledStateGraph


def draw_graph_png(filepath: str, graph: CompiledStateGraph):
    try:
        with open(filepath, "wb") as f:
            f.write(graph.get_graph().draw_mermaid_png())
    except Exception as e:
        print("Failed to draw graph cause by:", e)
