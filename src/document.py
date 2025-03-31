from dataclasses import dataclass
from typing import Any


@dataclass(init=True)
class Document:
    text: str
    metadata: dict[str, Any]
