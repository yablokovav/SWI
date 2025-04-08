from dataclasses import dataclass

@dataclass
class Message:
    is_error: bool
    is_warning: bool
    message: str
