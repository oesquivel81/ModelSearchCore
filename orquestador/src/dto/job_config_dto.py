from pydantic import BaseModel, Field
from typing import Any, Dict
from dto.model_type import ModelType
from dto.job_status import JobStatus


class JobConfigDTO(BaseModel):
    model_type: ModelType
    hyperparameters: Dict[str, Any] = Field(default_factory=dict)
    status: JobStatus = JobStatus.QUEUED
