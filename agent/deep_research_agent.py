from dotenv import load_dotenv

from agent.deep_researcher.graph import graph
from agent.shared.utils import draw_graph_png

load_dotenv()


draw_graph_png("deep_research_agent.png", graph)
stream = graph.stream({"research_topic": "kubernetes"})
for s in stream:
    for agent, e in s.items():
        print("===========================")
        print(agent, e, flush=True)
        print("===========================")
