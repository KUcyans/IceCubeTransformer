from enum import Enum
class LrDecayMode(Enum):
    LINEAR = 'linear'
    EXPONENTIAL = 'exponential'
    COSINE = 'cosine'
    
    @staticmethod
    def from_str(value: str) -> 'LrDecayMode':
        """Get the corresponding LrDecay by string value."""
        for lr_decay in LrDecayMode:
            if lr_decay.value == value:
                return lr_decay
        return None
    
    @staticmethod
    def to_str(lr_decay: 'LrDecayMode') -> str:
        """Get the string value of the LrDecay."""
        return lr_decay.value