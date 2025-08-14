from pydantic import BaseModel, Field
from typing import List, Optional


class InferenceRequest(BaseModel):
    inputs: List[float] = Field(
        ..., description="Flattened list of input values for the model"
    )
    model_name: str = Field(..., description="Name of the Triton model to use")
    model_version: Optional[str] = Field(
        None, description="Specific model version (defaults to latest)"
    )


class InferenceResponse(BaseModel):
    outputs: List[float] = Field(
        ..., description="Flattened list of output values from the model"
    )
    model_name: str = Field(..., description="Name of the model used for inference")
    model_version: str = Field(..., description="Version of the model used")
