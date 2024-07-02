from fastapi import APIRouter # type: ignore
from api.endpoints import endpoints


api_router = APIRouter()
api_router.include_router(endpoints.router)
