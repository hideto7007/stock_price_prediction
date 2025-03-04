from pydantic import BaseModel


###########################
# バリデーションモデル #
###########################

class ValidatonModel(BaseModel):
    field: str
    message: str
