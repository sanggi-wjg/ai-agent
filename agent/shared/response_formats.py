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
