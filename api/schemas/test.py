from typing import Literal
from pydantic import BaseModel, Field


class RequestId(BaseModel):
    request_id: str = Field(
        default="3f77bd7a-77a6-44b1-b4bc-a84eef8cced2",
        description="リクエストID"
    )


class TestRequest(BaseModel):
    state: RequestId
    method: Literal["GET", "POST", "PUT", "DELETE"]
    headers: dict = Field(default_factory=dict, description="リクエストヘッダー")
    params: dict = Field(default_factory=dict, description="リクエストパラメータ")
    url: str = Field(default="http://test", description="リクエストURL")
