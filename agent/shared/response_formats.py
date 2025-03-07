from pydantic import BaseModel, Field


class ResearchAgentResponse(BaseModel):
    """Schema for an agent's response. This model represents the response structure for an AI agent."""

    answer: str = Field(description="Answer to the question")
    source: str = Field(description="Source of the information")
