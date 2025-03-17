import re
from typing import Dict, Any, Optional

import requests
import yaml
from langchain_community.agent_toolkits.openapi.spec import ReducedOpenAPISpec
from langchain_core.utils.json_schema import dereference_refs
from langgraph.graph.state import CompiledStateGraph

from agent.shared.response_formats import APIPlanResponse


def draw_graph_png(filepath: str, graph: CompiledStateGraph):
    try:
        with open(filepath, "wb") as f:
            f.write(graph.get_graph().draw_mermaid_png())
    except Exception as e:
        print("Failed to draw graph cause by:", e)


def clean_deepseek_chat_response(chat_response: str):
    return re.sub(r'<think>.*?</think>', '', chat_response, flags=re.DOTALL).strip()


def reduce_openapi_spec(
    filepath: str,
    target_server_env: str = None,
    target_tags: list[str] = None,
    dereference: bool = True,
) -> ReducedOpenAPISpec:
    with open(filepath, "r") as f:
        json_api_spec = yaml.safe_load(f)
    return _reduce_my_openapi_spec(json_api_spec, target_server_env, target_tags, dereference)


def _reduce_my_openapi_spec(
    spec: dict,
    target_server_env: str = None,
    target_tags: list[str] = None,
    dereference: bool = True,
) -> ReducedOpenAPISpec:
    target_server_env = target_server_env or "dev"
    target_tags = set(tag.lower() for tag in target_tags) or set()
    target_methods = ["get", "post", "patch", "put", "delete"]

    endpoints = []
    for route, operation in spec["paths"].items():
        for operation_name, docs in operation.items():
            ok_method = operation_name.lower() in target_methods
            ok_tag = set(tag.lower() for tag in docs.get("tags")).intersection(target_tags) != set()

            if ok_method and ok_tag:
                endpoints.append((f"{operation_name.upper()} {route}", docs.get("description"), docs))

    if dereference:
        endpoints = [
            (name, description, dereference_refs(docs, full_schema=spec)) for name, description, docs in endpoints
        ]

    def reduce_endpoint_docs(docs: dict) -> dict:
        out = {}
        if docs.get("description"):
            out["description"] = docs.get("description")
        if docs.get("parameters"):
            out["parameters"] = [parameter for parameter in docs.get("parameters", []) if parameter.get("required")]
        if "200" in docs["responses"]:
            out["responses"] = docs["responses"]["200"]
        if docs.get("requestBody"):
            out["requestBody"] = docs.get("requestBody")
        return out

    def filter_servers(servers: list[dict]) -> list[dict]:
        servers = [server for server in servers if target_server_env in server["url"]]
        if not servers:
            raise ValueError(f"Server {target_server_env} not found in {spec['servers']}")
        return servers

    endpoints = [(name, description, reduce_endpoint_docs(docs)) for name, description, docs in endpoints]
    return ReducedOpenAPISpec(
        servers=filter_servers(spec["servers"]),
        description="",
        endpoints=endpoints,
    )


def request_api_by_plan(server: str, plan: APIPlanResponse, token: Optional[str]) -> Dict[str, Any]:
    try:
        url = server + plan.endpoint
        response = requests.request(
            method=plan.method,
            url=url,
            params=plan.query_params,
            json=plan.payload,
            headers={"Authorization": f"Bearer {token}"} if token else {},
        )
        return {
            "endpoint": plan.endpoint,
            "is_success": response.ok,
            "status_code": response.status_code,
            "body": response.json() if response.headers.get("Content-Type") == "application/json" else response.text,
        }
    except Exception as e:
        return {
            "error": str(e),
        }


def escape_with_double_curly_braces(element: Any) -> str:
    return str(element).replace("{", "{{").replace("}", "}}")
