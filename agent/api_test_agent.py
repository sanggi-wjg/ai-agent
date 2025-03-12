from typing import Literal

from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langgraph.constants import START, END
from langgraph.graph import StateGraph
from langgraph.types import Command

from agent.shared.prompts import API_TEST_PLAN_INSTRUCTIONS
from agent.shared.response_formats import APIPlanResponse
from agent.shared.states import APITestState
from agent.shared.utils import draw_graph_png, request_api_by_plan, reduce_openapi_spec


class Node:
    REQUEST_NODE = "request_node"
    PLAN_NODE = "plan_node"
    FINALIZE_NODE = "finalize_node"


def plan_node(state: APITestState):
    endpoint = str(state.open_api_spec.endpoints[state.endpoint_index]).replace("{", "{{").replace("}", "}}")
    success_results = [res for res in state.request_results if res["is_success"]]

    system_message = API_TEST_PLAN_INSTRUCTIONS + "<API_SPECIFICATION>" + endpoint + "</API_SPECIFICATION>"
    if success_results:
        system_message += "<Previous Success Results>\n" + str(success_results) + "\n</Previous Success Results>"

    llm = ChatOllama(model="qwen2.5:14b-instruct-q8_0", temperature=0.1).with_structured_output(APIPlanResponse)
    prompt = ChatPromptTemplate.from_messages(
        [
            {"role": "system", "content": system_message},
        ]
    )
    chain = prompt | llm
    chat_response = chain.invoke({"input": "Plan an API request"})
    return Command(
        goto=Node.REQUEST_NODE,
        update={
            "request_plans": state.request_plans + [chat_response],
        },
    )


def request_node(state: APITestState):
    server = state.open_api_spec.servers[0]["url"]
    result = request_api_by_plan(server, state.request_plans[-1], state.token)
    return {
        "endpoint_index": state.endpoint_index + 1,
        "request_results": state.request_results + [result],
    }


def finalize_node(state: APITestState):
    llm = ChatOllama(model="exaone3.5:7.8b-instruct-fp16", temperature=0.1)
    prompt = ChatPromptTemplate.from_messages(
        [
            {
                "role": "system",
                "content": "You are a summarizer. Your task is to summarize the results of an API request.",
            },
            {"role": "user", "content": "Summarize the results in KOREAN: {results}"},
        ]
    )
    chain = prompt | llm
    chat_response = chain.invoke({"results": state.request_results})
    return Command(
        goto=END,
        update={
            "summary": chat_response.content,
        },
    )


def router(state: APITestState) -> Literal["plan_node", "finalize_node"]:
    if state.endpoint_index < state.endpoint_size:
        return "plan_node"

    return "finalize_node"


graph_builder = StateGraph(APITestState)
graph_builder.add_node(Node.PLAN_NODE, plan_node)
graph_builder.add_node(Node.REQUEST_NODE, request_node)
graph_builder.add_node(Node.FINALIZE_NODE, finalize_node)

graph_builder.add_edge(START, Node.PLAN_NODE)
graph_builder.add_edge(Node.PLAN_NODE, Node.REQUEST_NODE)
graph_builder.add_conditional_edges(Node.REQUEST_NODE, router)
graph_builder.add_edge(Node.FINALIZE_NODE, END)
graph = graph_builder.compile()
draw_graph_png("api_test_agent.png", graph)

tag_input = "banner"
server_env_input = "stg"
access_token_input = None

open_api_spec = reduce_openapi_spec(
    filepath="dataset/donotcommit.yaml",
    target_server_env=server_env_input,
    target_tags=[tag_input],
    dereference=True,
)
if not open_api_spec.endpoints:
    print("No endpoints found")
    exit()

stream = graph.stream(
    {
        "token": access_token_input,
        "open_api_spec": open_api_spec,
        "endpoint_size": len(open_api_spec.endpoints),
        "endpoint_index": 0,
        "request_plans": [],
        "request_results": [],
    }
)
for s in stream:
    for agent, res in s.items():
        print(agent, res, flush=True)
