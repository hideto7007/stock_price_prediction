from pydantic.generics import GenericModel
# from datetime import datetime
from typing import Generic, Optional, TypeVar, List

# ジェネリック型 T を定義
T = TypeVar("T")


class Content(GenericModel, Generic[T]):
    # 共通のモデルを定義
    result: Optional[List[T] | T]
