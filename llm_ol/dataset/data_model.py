from pydantic import BaseModel


class Category(BaseModel):
    id_: str
    name: str
    children: list[str] = []
