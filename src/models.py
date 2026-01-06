"""
Pydantic models for structured input/output with LLMs.
"""
from pydantic import BaseModel, Field


class SyllogismInput(BaseModel):
    """Structured representation of a syllogism."""
    major_premise: str = Field(description="The major premise of the syllogism")
    minor_premise: str = Field(description="The minor premise of the syllogism")
    conclusion: str = Field(description="The conclusion of the syllogism")


class LLMResponse(BaseModel):
    """Structured response from the LLM."""
    validity: bool = Field(description="Whether the conclusion logically follows from the premises")


class ValidityPrediction(BaseModel):
    """Final prediction format for submission."""
    id: str = Field(description="Unique identifier for the syllogism")
    validity: bool = Field(description="Predicted validity (true/false)")


class TrainingExample(BaseModel):
    """Training data example."""
    id: str
    syllogism: str
    validity: bool
    plausibility: bool


class TestExample(BaseModel):
    """Test data example."""
    id: str
    syllogism: str
