from typing import Literal

from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langgraph.constants import START, END
from langgraph.graph import StateGraph
from tavily import TavilyClient

from agent.shared.prompts import QUERY_WRITER_INSTRUCTIONS, SUMMARIZE_INSTRUCTIONS, REFLECTION_INSTRUCTIONS
from agent.shared.response_formats import QueryWriterResponse, ReflectionResponse
from agent.shared.states import DeepResearchState
from agent.shared.utils import clean_deepseek_chat_response, draw_graph_png

load_dotenv()


def generate_query_node(state: DeepResearchState):
    llm = ChatOllama(model="qwen2.5:14b-instruct-q8_0", temperature=0.1).with_structured_output(QueryWriterResponse)
    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessage(
                content=QUERY_WRITER_INSTRUCTIONS.format(topic=state["research_topic"]),
            ),
        ]
    )
    chain = prompt | llm
    chat_response = chain.invoke({"input": "Generate a query for web search:"})
    return {
        "research_query": chat_response.query,
    }


def web_search_node(state: DeepResearchState):
    tavily_client = TavilyClient()
    search_response = tavily_client.search(
        query=state["research_query"],
        max_results=1,
        include_raw_content=True,
    )
    return {
        "web_search_loop_count": state["web_search_loop_count"] + 1,
        "web_search_responses": state["web_search_responses"] + [search_response],
    }


def summarize_source_node(state: DeepResearchState):
    research_topic = state["research_topic"]
    summary = state["summary"]
    web_search_response = state["web_search_responses"][-1]

    if summary:
        human_message = (
            f"<User Input>\n{research_topic}\n<User Input>\n\n"
            f"<Existing Summary>\n{summary}\n<Existing Summary>\n\n"
            f"<New Search Results>\n{web_search_response}\n<New Search Results>"
        )
    else:
        human_message = (
            f"<User Input>\n{research_topic}\n<User Input>\n\n"
            f"<Search Results>\n{web_search_response}\n<Search Results>"
        )

    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessage(content=SUMMARIZE_INSTRUCTIONS),
            HumanMessage(content=human_message),
        ]
    )
    llm = ChatOllama(model="deepseek-r1:14b", temperature=0.1)
    chain = prompt | llm
    chat_response = chain.invoke({"input": research_topic})
    return {
        "summary": clean_deepseek_chat_response(chat_response.content),
    }


def reflect_on_summary_node(state: DeepResearchState):
    llm = ChatOllama(model="qwen2.5:14b-instruct-q8_0", temperature=0.2).with_structured_output(ReflectionResponse)
    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessage(content=REFLECTION_INSTRUCTIONS.format(topic=state["research_topic"])),
            HumanMessage(
                content="Identify a knowledge gap and generate a follow-up web search query based on our existing knowledge: {summary}"
            ),
        ]
    )
    chain = prompt | llm
    chat_response = chain.invoke({"summary": state["summary"]})
    return {
        "search_query": chat_response.follow_up_query,
        "keep_searching": chat_response.keep_searching,
    }


def finalize_summary_node(state: DeepResearchState):
    # tools = [write_file_tool]
    # llm = ChatOllama(model="qwen2.5:14b-instruct-q8_0", temperature=0.3).bind_tools(tools)
    # prompt = ChatPromptTemplate.from_messages(
    #     [
    #         SystemMessage(content="Write a file of the following text"),
    #         HumanMessage(content=f"Finalize the summary: {state['summary']}"),
    #     ]
    # )
    return {
        "summary": state["summary"],
    }


def router(state: DeepResearchState) -> Literal["finalize_summary_node", "web_search_node"]:
    if state["web_search_loop_count"] <= state["max_web_search_loop_count"]:
        return "finalize_summary_node"

    if state["keep_searching"]:
        return "web_search_node"

    return "finalize_summary_node"


graph_builder = StateGraph(DeepResearchState)
graph_builder.add_node("generate_query_node", generate_query_node)
graph_builder.add_node("web_search_node", web_search_node)
graph_builder.add_node("summarize_source_node", summarize_source_node)
graph_builder.add_node("reflect_on_summary_node", reflect_on_summary_node)
graph_builder.add_node("finalize_summary_node", finalize_summary_node)

graph_builder.add_edge(START, "generate_query_node")
graph_builder.add_edge("generate_query_node", "web_search_node")
graph_builder.add_edge("web_search_node", "summarize_source_node")
graph_builder.add_edge("summarize_source_node", "reflect_on_summary_node")
graph_builder.add_conditional_edges("reflect_on_summary_node", router)
graph_builder.add_edge("finalize_summary_node", END)

graph = graph_builder.compile()
draw_graph_png("deep_research_agent.png", graph)

stream = graph.stream(
    {
        "research_topic": input("조사 주제:"),
        "web_search_loop_count": 0,
        "max_web_search_loop_count": 2,
        "web_search_responses": [],
        "keep_searching": False,
        "summary": None,
    }
)
for s in stream:
    for agent, e in s.items():
        print(agent, e, flush=True)
