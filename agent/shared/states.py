from dataclasses import dataclass, field
from typing import TypedDict, List

from langchain_community.agent_toolkits.openapi.spec import ReducedOpenAPISpec

from agent.shared.response_formats import APIPlanResponse


class DeepResearchState(TypedDict):
    research_topic: str
    research_query: str
    web_search_loop_count: int
    max_web_search_loop_count: int
    web_search_responses: list
    keep_searching: bool
    summary: str


@dataclass(kw_only=True)
class APITestState:
    token: str = field(default=None)
    open_api_spec: ReducedOpenAPISpec = field(default=None)
    endpoint_size: int = field(default=0)
    endpoint_index: int = field(default=0)
    request_plans: List[APIPlanResponse] = field(default=None)
    request_results: List[dict] = field(default=None)
    summary: str = field(default=None)
