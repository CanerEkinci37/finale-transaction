from fastapi import APIRouter, HTTPException

from ... import crud
from ...models import ConfigRead, ConfigUpdate
from ..deps import SessionDep

router = APIRouter()


@router.get("/", response_model=ConfigRead | None)
async def get_config(*, session: SessionDep):
    config = crud.get_config(session=session)
    if config is None:
        raise HTTPException(status_code=404, detail="Config not found")
    return config


@router.put("/", response_model=ConfigRead | None)
async def set_config(*, session: SessionDep, config_update: ConfigUpdate):
    return crud.set_config(session=session, config_update=config_update)
