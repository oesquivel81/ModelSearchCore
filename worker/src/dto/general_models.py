from pydantic import BaseModel, Field
from typing import Optional, Any, Dict
from datetime import datetime
from dto.model_type import ModelType


class GeneralModels(BaseModel):
    model_id: str
    model_type: ModelType
    hyperparameters: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    status: Optional[str] = None

    class Config:
        from_attributes = True
