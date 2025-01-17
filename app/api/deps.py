from typing import Annotated, Generator

from fastapi import Depends
from sqlmodel import Session

from ..core.db import engine


def get_db() -> Generator[Session, None, None]:
    """Dependency for database session."""
    with Session(engine) as session:
        yield session


SessionDep = Annotated[Session, Depends(get_db)]
