from typing import List, Optional, Any

from pydantic import BaseModel, Field


class ResearchAgentResponse(BaseModel):
    """Schema for an agent's response. This model represents the response structure for an AI agent."""

    answer: str = Field(description="Answer to the question")
    source: str = Field(description="Source of the information")


class QueryWriterResponse(BaseModel):
    """Schema for a query writer's response. This model represents the response structure for a query writer."""

    query: str = Field(description="The actual search query string")
    aspect: str = Field(description="The specific aspect of the topic being researched")
    rationale: str = Field(description="Brief explanation of why this query is relevant")


class ReflectionResponse(BaseModel):
    """Schema for a reflection response. This model represents the response structure for a reflection."""

    knowledge_gap: str = Field(description="Describe what information is missing or needs clarification")
    follow_up_query: str = Field(description="Write a specific question to address this gap")
    keep_searching: bool = Field(description="Should the agent keep searching for more information?")


class APIPlanResponse(BaseModel):
    """Schema for an API test request plan.

    This model defines the structure for an API test plan, including the request method,
    endpoint, and optional parameters or payload.
    """

    method: str = Field(
        description="The HTTP method to use for the request.",
        enum=["GET", "POST", "PATCH", "PUT", "DELETE"],
    )
    endpoint: str = Field(description="The target API endpoint to be tested.")
    query_params: Optional[dict] = Field(
        description="A dictionary of query parameters to be sent with the request. If None, no query parameters will be included.",
        default=None,
    )
    payload: Optional[Any] = Field(
        description="The request payload (body) as a dictionary or list. If None, no payload will be sent.",
        default=None,
    )
