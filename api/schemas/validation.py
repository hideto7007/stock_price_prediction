from pydantic import BaseModel  # type: ignore


###########################
# バリデーションモデル #
###########################

class ValidatonModel(BaseModel):
    field: str
    message: str
