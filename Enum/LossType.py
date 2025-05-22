from enum import Enum

class LossType(Enum):
    MSE          = (0, 'mse', "mean_squared_error")
    CROSSENTROPY = (1, 'ce', "cross_entropy")
    TAUPURITYMSE = (2, 'tau', "mse wih tau_purity term")
    
    def __init__(self, value, alias, description):
        self._value_ = value
        self._alias_ = alias
        self._description_ = description

    @classmethod
    def from_string(cls, string):
        for loss_type in cls:
            if loss_type.alias == string or loss_type.description == string:
                return loss_type
        raise ValueError(f"Invalid loss type: {string}")
    
    @classmethod
    def from_value(cls, value):
        for loss_type in cls:
            if loss_type.value == value:
                return loss_type
        raise ValueError(f"Invalid loss type value: {value}")
    
    @property
    def value(self):
        return self._value_
    @property
    def alias(self):
        return self._alias_
    @property
    def description(self):
        return self._description_