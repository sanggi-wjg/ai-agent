from typing import TypedDict


class DeepResearchState(TypedDict):
    research_topic: str
    research_query: str
    web_search_loop_count: int
    max_web_search_loop_count: int
    web_search_responses: list
    keep_searching: bool
    summary: str
