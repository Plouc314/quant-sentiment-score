from datetime import date
from typing import TypedDict


class Story(TypedDict):
    id: str
    url: str
    title: str
    publish_date: date
    source_name: str
    language: str


class Article(TypedDict):
    id: str
    url: str
    title: str
    text: str
    publish_date: date
    source_name: str
    language: str
