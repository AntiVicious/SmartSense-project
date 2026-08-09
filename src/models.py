"""SQLAlchemy ORM models and Pydantic request/response schemas."""

from sqlalchemy import Column, Float, Integer, String, Text

from pydantic import BaseModel

from .db import Base


class Property(Base):
    __tablename__ = "properties"
    id = Column(Integer, primary_key=True, index=True)
    title = Column(String, index=True)
    description = Column(Text)
    location = Column(String)
    price = Column(Float)
    listing_date = Column(String)
    certifications_link = Column(String)
    floorplan_image_url = Column(String)
    rooms = Column(Integer)
    halls = Column(Integer)
    kitchens = Column(Integer)
    bathrooms = Column(Integer)


class ChatRequest(BaseModel):
    query: str
    history: list[tuple[str, str]] = []
