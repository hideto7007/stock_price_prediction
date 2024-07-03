from fastapi import APIRouter # type: ignore
from api.endpoints import stock_price


api_router = APIRouter()
api_router.include_router(stock_price.router)
