from sqlmodel import Session, select

from .models import Config, ConfigCreate, ConfigUpdate


def create_config(*, session: Session, config_create: ConfigCreate):
    db_obj = Config.model_validate(config_create)
    session.add(db_obj)
    session.commit()
    session.refresh(db_obj)
    return db_obj


def get_config(*, session: Session):
    statement = select(Config)
    session_config = session.exec(statement).first()
    return session_config


def set_config(*, session: Session, config_update: ConfigUpdate):
    config_data = config_update.model_dump()

    statement = select(Config)
    db_config = session.exec(statement).first()
    db_config.sqlmodel_update(config_data)
    session.add(db_config)
    session.commit()
    session.refresh(db_config)
    return db_config
