import re

from langgraph.graph.state import CompiledStateGraph


def draw_graph_png(filepath: str, graph: CompiledStateGraph):
    try:
        with open(filepath, "wb") as f:
            f.write(graph.get_graph().draw_mermaid_png())
    except Exception as e:
        print("Failed to draw graph cause by:", e)


def clean_deepseek_chat_response(chat_response: str):
    return re.sub(r'<think>.*?</think>', '', chat_response, flags=re.DOTALL).strip()
