import logging
import time

from fastapi import APIRouter, HTTPException, Query

from ... import crud, train
from ...utils import utils
from ..deps import SessionDep

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("API")

router = APIRouter()


@router.get("/")
async def get_train(
    *, session: SessionDep, models: list[str] | None = Query(default=None)
):
    started_at = time.time()

    # Get Configuration
    config = crud.get_config(session=session)
    if config is None:
        raise HTTPException(status_code=404, detail="Config not found")

    try:
        # Call the prediction function with the config and models list
        train.start(config=config.data, models=models)

        # Prepare and return the results
        return {
            "status": "Training is completed successfully",
            "estimated_time": utils.format_time(time.time() - started_at),
        }
    except (IndexError, KeyError, ValueError, TypeError) as specific_error:
        logger.error(f"Known error countered: {specific_error}")
        raise HTTPException(status_code=400, detail=f"Invalid input: {specific_error}")
    except Exception as e:
        logger.exception("Unexpected error during training")
        raise HTTPException(
            status_code=500, detail=f"An unexpected error occurred: {str(e)}"
        )
