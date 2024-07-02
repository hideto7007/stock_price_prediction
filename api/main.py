from fastapi import FastAPI # type: ignore
from api.router.router import api_router


app = FastAPI()
app.include_router(api_router)
