from pydantic import BaseModel
import typing

class TokenLogprob(BaseModel):
    token_id: int
    text: str
    logprob: float
    prob_percent: float

class GenerationResponse(BaseModel):
    full_text: str
    tokens: typing.List[TokenLogprob]

class ModelInfo(BaseModel):
    id: str
    path: str

class ModelLoadRequest(BaseModel):
    model_id: str

class ModelStatusResponse(BaseModel):
    active_model_id: typing.Optional[str]
    is_loaded: bool
