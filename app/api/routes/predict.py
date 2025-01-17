import logging
import time

from fastapi import APIRouter, Body, HTTPException

from ... import crud, predict
from ...utils import utils
from ..deps import SessionDep

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("API")

router = APIRouter()


@router.post("/")
async def get_predict(*, session: SessionDep, data: list[dict] = Body()):
    started_at = time.time()

    # Get configuration
    config = crud.get_config(session=session)
    if config is None:
        raise HTTPException(status_code=404, detail="Config not found")

    try:
        # Call the prediction function with the config and input data
        predictions = predict.start(config=config.data, test_data=data)

        # Prepare and return the results
        return {
            "status": "Prediction completed successfully",
            "predictions": predictions,  # Return the predictions here
            "estimated_time": utils.format_time(time.time() - started_at),
        }
    except (IndexError, KeyError, ValueError, TypeError) as specific_error:
        logger.error(f"Known error encountered: {specific_error}")
        raise HTTPException(status_code=400, detail=f"Invalid input: {specific_error}")
    except Exception as e:
        logger.exception("Unexpected error during training")
        raise HTTPException(
            status_code=500, detail=f"An unexpected error occurred: {str(e)}"
        )
