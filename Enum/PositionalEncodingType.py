from enum import Enum

class PositionalEncodingType(Enum):
    ABSOLUTE = (0, "absolute", "Absolute Positional Encoding")
    ROPE = (1, "rope", "Rotary Positional Encoding")
    T5 = (2, "T5", "Text-to-Text Transfer Transformer")
    ALIBI = (3, "alibi", "Attention with LInear BIas")
    
    def __init__(self, value, name, description):
        self._value_ = value
        self._name_ = name
        self._description_ = description
    @classmethod
    def from_string(cls, string: str):
        for mode in cls:
            if mode.name == string:
                return mode
        raise ValueError(f"Invalid attention type: {string}")
    @classmethod
    def from_value(cls, value: int):
        for mode in cls:
            if mode.value == value:
                return mode
        raise ValueError(f"Invalid attention type value: {value}")
    @property
    def value(self) -> int:
        return self._value_
    @property
    def name(self) -> str:
        return self._name_
    @property
    def description(self) -> str:
        return self._description_