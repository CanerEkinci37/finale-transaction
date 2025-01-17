from sqlmodel import Session, SQLModel, create_engine, select

from .. import crud
from ..core.config import settings
from ..models import Config, ConfigCreate

engine = create_engine(settings.DATABASE_URL)


def init_db(session: Session):
    SQLModel.metadata.create_all(engine)

    statement = select(Config)
    config = session.exec(statement).all()

    if len(config) < 1:
        config_create = ConfigCreate(data={})
        crud.create_config(session=session, config_create=config_create)
