from fastapi import APIRouter

from .routes import config, predict, train

api_router = APIRouter()
api_router.include_router(config.router, prefix="/config", tags=["config"])
api_router.include_router(train.router, prefix="/train", tags=["train"])
api_router.include_router(predict.router, prefix="/predict", tags=["predict"])
