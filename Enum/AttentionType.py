from enum import Enum

class AttentionType(Enum):
    SDP = (0, "scaled_dot", "torch.nn.functional.scaled_dot_product_attention")
    INNOCENT = (1, "innocent", "Explicit tensor product")
    T5 = (2, "t5", "Text-to-Text Transfer Transformer")
    ALIBI = (3, "alibi", "Attention with LInear BIas")
    XFORMERS = (4, "xformers", "XFormers library")
    
    def __init__(self, value: int, name: str, description: str):
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