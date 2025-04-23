from enum import Enum
class PositionalEncodingType(Enum):
    ABSOLUTE = 0
    ROPE = 1
    T5 = 2
    ALIBI = 3

    @classmethod
    def from_string(cls, string: str):
        string = string.lower()
        name_map = {
            "absolute": cls.ABSOLUTE,
            "rope": cls.ROPE,
            "t5": cls.T5,
            "alibi": cls.ALIBI,
        }
        if string in name_map:
            return name_map[string]
        raise ValueError(f"Invalid positional encoding type: {string}")