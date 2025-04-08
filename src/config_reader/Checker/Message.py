from dataclasses import dataclass

@dataclass
class Message:
    is_error: bool
    message: str
