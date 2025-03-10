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
