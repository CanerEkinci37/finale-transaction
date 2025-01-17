from sqlalchemy.dialects.postgresql import JSON
from sqlmodel import Column, Field, SQLModel


class ConfigBase(SQLModel):
    data: dict = Field(default_factory=dict, sa_column=Column(JSON))


class ConfigCreate(ConfigBase):
    pass


class ConfigRead(ConfigBase):
    pass


class ConfigUpdate(ConfigBase):
    pass


class Config(ConfigBase, table=True):
    id: int = Field(default=1, primary_key=True)

    class Config:
        arbitrary_types_allowed = True
